let documents = [];
let pollInterval = null;
let liveTimerInterval = null;
let prevStatuses = {};
let selectedDocumentIds = [];
let sessionQuestions = [];

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('active'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('active'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => { handleFiles(e.target.files); e.target.value = ''; });

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) sendMessage(); });

    loadDocuments();
    loadStats();
});

async function handleFiles(files) {
    const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (!pdfs.length) { showToast('Please upload PDF files only.', 'error'); return; }

    const formData = new FormData();
    pdfs.forEach(f => formData.append('files', f));

    showToast(`Uploading ${pdfs.length} PDF(s)...`, 'info');

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        loadDocuments();
        startPolling();
    } catch (error) {
        showToast('Upload failed: ' + error.message, 'error');
    }
}

function startPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
        const oldStatuses = { ...prevStatuses };
        await loadDocuments();

        documents.forEach(doc => {
            if (oldStatuses[doc.id] === 'processing' && doc.status === 'ready') {
                showToast(`"${doc.filename.split('.')[0]}" is ready — ask questions!`, 'success');
            }
            if (oldStatuses[doc.id] === 'processing' && doc.status === 'error') {
                showToast(`Failed to process "${doc.filename}"`, 'error');
            }
        });

        const hasProcessing = documents.some(d => d.status === 'processing');
        if (!hasProcessing) {
            clearInterval(pollInterval);
            pollInterval = null;
            stopLiveTimer();
            loadStats();
        }
    }, 3000);
}

function startLiveTimer() {
    if (liveTimerInterval) return;
    liveTimerInterval = setInterval(tickLiveTimers, 1000);
}

function stopLiveTimer() {
    if (liveTimerInterval) {
        clearInterval(liveTimerInterval);
        liveTimerInterval = null;
    }
}

function tickLiveTimers() {
    const now = Date.now();
    document.querySelectorAll('.time-label[data-start]').forEach(el => {
        const start = parseInt(el.dataset.start, 10);
        el.textContent = formatDuration((now - start) / 1000);
    });
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        documents = data.documents;
        prevStatuses = {};
        documents.forEach(d => { prevStatuses[d.id] = d.status; });
        renderDocuments();
        updateHeaderBadge();
        updatePdfStats();

        if (documents.some(d => d.status === 'processing')) {
            startLiveTimer();
        } else {
            stopLiveTimer();
        }
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

function parseSQLiteDate(s) {
    if (!s) return null;
    return new Date(s.replace(' ', 'T'));
}

function formatDuration(seconds) {
    seconds = Math.floor(seconds);
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}m ${s}s`;
}

function updatePdfStats() {
    const total = documents.length;
    const ready = documents.filter(d => d.status === 'ready');

    document.getElementById('totalPdfs').textContent = total;

    let totalSeconds = 0;
    let allTimed = ready.length > 0;
    ready.forEach(doc => {
        const start = parseSQLiteDate(doc.upload_date);
        const end = parseSQLiteDate(doc.completed_at);
        if (start && end) {
            totalSeconds += (end - start) / 1000;
        } else {
            allTimed = false;
        }
    });

    const timeEl = document.getElementById('totalTime');
    if (ready.length === 0) {
        timeEl.textContent = '—';
    } else {
        timeEl.textContent = formatDuration(totalSeconds);
    }
}

function updateHeaderBadge() {
    const ready = documents.filter(d => d.status === 'ready').length;
    const processing = documents.filter(d => d.status === 'processing').length;
    const errors = documents.filter(d => d.status === 'error').length;
    const badge = document.getElementById('headerBadge');
    const summary = document.getElementById('docSummary');

    const parts = [];
    if (ready) parts.push(`${ready} ready`);
    if (processing) parts.push(`${processing} processing`);
    if (errors) parts.push(`${errors} failed`);
    summary.textContent = parts.length ? parts.join(' · ') : 'No documents yet';

    if (processing > 0) {
        badge.textContent = `${processing} processing...`;
        badge.className = 'badge processing';
    } else if (ready > 0) {
        badge.textContent = `${ready} doc${ready > 1 ? 's' : ''} ready`;
        badge.className = 'badge ready';
    } else {
        badge.textContent = 'No documents';
        badge.className = 'badge';
    }

    updateSendButtonState();
}

function updateSendButtonState() {
    const processing = documents.filter(d => d.status === 'processing').length;
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');
    if (!sendBtn || !chatInput) return;

    if (processing > 0) {
        sendBtn.disabled = true;
        sendBtn.title = 'Waiting for PDF to finish processing…';
        chatInput.placeholder = 'Waiting for PDF to be ready…';
    } else {
        // Only re-enable if sendMessage didn't already disable it for a pending request
        if (sendBtn.dataset.waiting !== 'true') {
            sendBtn.disabled = false;
            sendBtn.title = '';
        }
        chatInput.placeholder = 'Ask a question or say hi…';
    }
}

function renderDocuments() {
    const list = document.getElementById('fileList');
    if (!documents.length) {
        list.innerHTML = '<div class="empty-docs">Upload PDFs to get started</div>';
        return;
    }

    const statusIcon = { ready: '✓', processing: '⟳', error: '✗' };
    const statusLabel = { ready: 'Ready', processing: 'Processing...', error: 'Failed' };
    const now = Date.now();

    let html = '';

    const readyDocs = documents.filter(d => d.status === 'ready');
    const allSelected = readyDocs.length > 0 && readyDocs.every(d => selectedDocumentIds.includes(d.id));

    if (readyDocs.length > 1) {
        html += `<div class="all-docs-btn select-all-btn" onclick="toggleSelectAll()">
            ${allSelected ? '☐ Deselect all' : `☑ Select all (${readyDocs.length})`}
        </div>`;
    }

    html += documents.map(doc => {
        const pages = doc.page_count ? ` · ${doc.page_count}p` : '';
        let timeHtml = '';
        const isSelected = selectedDocumentIds.includes(doc.id);

        if (doc.status === 'processing') {
            const start = parseSQLiteDate(doc.upload_date);
            const startMs = start ? start.getTime() : now;
            const elapsed = formatDuration((now - startMs) / 1000);
            timeHtml = `<span class="time-label" data-start="${startMs}">${elapsed}</span>`;
        } else if (doc.status === 'ready' && doc.completed_at) {
            const start = parseSQLiteDate(doc.upload_date);
            const end = parseSQLiteDate(doc.completed_at);
            if (start && end) {
                timeHtml = `<span class="time-label done">in ${formatDuration((end - start) / 1000)}</span>`;
            }
        } else if (doc.status === 'error' && doc.completed_at) {
            const start = parseSQLiteDate(doc.upload_date);
            const end = parseSQLiteDate(doc.completed_at);
            if (start && end) {
                timeHtml = `<span class="time-label err">after ${formatDuration((end - start) / 1000)}</span>`;
            }
        }

        const selectedClass = isSelected ? ' selected' : '';
        const checkboxHtml = doc.status === 'ready'
            ? `<input type="checkbox" class="doc-checkbox" ${isSelected ? 'checked' : ''} onchange="toggleDocument(${doc.id}, this.checked)" onclick="event.stopPropagation()">`
            : `<input type="checkbox" class="doc-checkbox" disabled>`;

        return `
        <div class="file-item ${doc.status}${selectedClass}">
            ${checkboxHtml}
            <span class="status-icon">${statusIcon[doc.status] || '?'}</span>
            <div class="file-info">
                <span class="name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
                <span class="status-label">${statusLabel[doc.status] || doc.status}${pages}${timeHtml ? ' · ' : ''}${timeHtml}</span>
            </div>
            <span class="delete" onclick="deleteDocument(${doc.id}, event)" title="Delete">&times;</span>
        </div>`;
    }).join('');

    list.innerHTML = html;
    updateChatContext();
}

function toggleDocument(id, checked) {
    if (checked) {
        if (!selectedDocumentIds.includes(id)) {
            selectedDocumentIds.push(id);
        }
    } else {
        selectedDocumentIds = selectedDocumentIds.filter(i => i !== id);
    }
    renderDocuments();
}

function clearDocumentSelection() {
    selectedDocumentIds = [];
    renderDocuments();
}

function toggleSelectAll() {
    const readyDocs = documents.filter(d => d.status === 'ready');
    const allSelected = readyDocs.every(d => selectedDocumentIds.includes(d.id));
    selectedDocumentIds = allSelected ? [] : readyDocs.map(d => d.id);
    renderDocuments();
}

function updateChatContext() {
    const ctx = document.getElementById('chatContext');
    const input = document.getElementById('chatInput');
    if (selectedDocumentIds.length > 0) {
        const names = selectedDocumentIds
            .map(id => documents.find(d => d.id === id))
            .filter(Boolean)
            .map(d => d.filename);
        const display = names.length <= 2 ? names.join(' & ') : `${names[0]} + ${names.length - 1} more`;
        ctx.innerHTML = `<span class="scope-label">Chatting about:</span> <span class="doc-name">${escapeHtml(display)}</span> <button class="clear-btn" onclick="clearDocumentSelection()">Clear</button>`;
        ctx.classList.add('active');
        input.placeholder = `Ask about ${escapeHtml(display)}...`;
    } else {
        ctx.classList.remove('active');
        ctx.innerHTML = '';
        input.placeholder = 'Ask a question or say hi...';
    }
}

async function deleteDocument(id, e) {
    e.stopPropagation();
    try {
        await fetch(`/api/documents/${id}`, { method: 'DELETE' });
        selectedDocumentIds = selectedDocumentIds.filter(i => i !== id);
        showToast('Document deleted.', 'info');
        loadDocuments();
        loadStats();
    } catch (error) {
        showToast('Delete failed: ' + error.message, 'error');
    }
}

function isConversational(question) {
    const GREETINGS = new Set([
        'hi','hii','hiii','hello','hey','howdy','hiya','yo','sup',
        'good morning','good afternoon','good evening','good night',
        'how are you','how r u','whats up',"what's up",
        'thanks','thank you','ty','thx','bye','goodbye','ok','okay',
        'cool','nice','great','awesome','got it','sure','alright',
    ]);
    const ABOUT_YOU = [
        'what do you do','what can you do','what is your work',
        'what are you','who are you',
        'tell me about you','tell me about yourself',
        'about you','about yourself',
        'what are your capabilities','what can you help with',
        'what is your purpose','how do you work','what do you help with',
        'what is your job','what are you for','what do you help me with',
        'describe yourself','introduce yourself',
    ];
    const q = question.toLowerCase().trim().replace(/[!?.,]+$/, '');
    if (GREETINGS.has(q)) return true;
    return ABOUT_YOU.some(s => q === s || q.startsWith(s));
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question) return;

    const readyDocs = documents.filter(d => d.status === 'ready');
    if (readyDocs.length > 0 && selectedDocumentIds.length === 0 && !isConversational(question)) {
        showToast('Please select at least one PDF before asking a question.', 'error');
        return;
    }

    // Store question context for eval — will be enriched with sources after response
    const questionIndex = sessionQuestions.length;
    const userMsgId = addMessage('user', escapeHtml(question));
    sessionQuestions.push({ question, sources: [], document_ids: selectedDocumentIds.slice(), doc_names: [], msgId: userMsgId });
    updateEvalCountBadge();
    input.value = '';

    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    sendBtn.dataset.waiting = 'true';

    const thinkingId = addMessage('assistant', '<span class="thinking">Thinking...</span>');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : null }),
        });
        if (!response.ok) {
            throw new Error(`Server error ${response.status} — please try again`);
        }
        const data = await response.json();

        let html = escapeHtml(data.answer).replace(/\n/g, '<br>');

        // Store answer + sources + full context texts + trace_id for accurate re-evaluation
        sessionQuestions[questionIndex].answer = data.answer || '';
        sessionQuestions[questionIndex].trace_id = data.trace_id || null;
        sessionQuestions[questionIndex].context_texts = data.context_texts || [];
        if (data.sources && data.sources.length > 0) {
            sessionQuestions[questionIndex].sources = data.sources;
            const selectedDocs = documents.filter(d => selectedDocumentIds.includes(d.id));
            sessionQuestions[questionIndex].doc_names = selectedDocs.map(d => d.filename);
            html += buildCiteRow(data.sources);
        }

        if (data.chunks_used > 0) {
            html += `<div class="msg-meta">${data.entities_found} entities · ${data.chunks_used} chunks used</div>`;
        }

        // Embed eval placeholder inside the bubble so scores appear in the same section
        let reportId = null;
        if (data.context_texts && data.context_texts.length > 0 && !data.needs_selection) {
            reportId = 'eval-card-' + Date.now();
            html += `<div id="${reportId}" class="eval-inline-card"><span class="eval-pending">Scoring…</span></div>`;
        }

        updateMessage(thinkingId, html);

        if (reportId) {
            runAutoEval(question, data.answer, data.context_texts, data.trace_id || null, reportId);
        }
    } catch (error) {
        updateMessage(thinkingId, 'Error: ' + error.message);
    } finally {
        sendBtn.dataset.waiting = 'false';
        updateSendButtonState();
        input.focus();
    }
}

function addMessage(role, content) {
    const messages = document.getElementById('messages');
    const id = 'msg-' + Date.now() + Math.random();
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.id = id;
    div.innerHTML = `<div class="bubble">${content}</div>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return id;
}

function updateMessage(id, content) {
    const el = document.getElementById(id);
    if (el) {
        el.querySelector('.bubble').innerHTML = content;
        document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/graph/stats');
        const data = await response.json();
        document.getElementById('stats').textContent =
            `${data.nodes} entities · ${data.edges} edges · ${data.chunks} chunks · ${data.documents} docs`;
    } catch {
        document.getElementById('stats').textContent = 'Graph stats unavailable';
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/* ── Tab switching ── */
function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-btn-' + tab).classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    if (tab === 'eval') updateEvalSessionInfo();
}

/* ── Eval badge on tab button ── */
function updateEvalCountBadge() {
    const badge = document.getElementById('evalCountBadge');
    if (sessionQuestions.length > 0) {
        badge.textContent = sessionQuestions.length;
        badge.style.display = 'inline-block';
    } else {
        badge.style.display = 'none';
    }
}

function updateEvalSessionInfo() {
    const n = sessionQuestions.length;
    document.getElementById('evalSessionInfo').textContent =
        n === 0 ? '0 questions asked this session'
                : `${n} question${n > 1 ? 's' : ''} asked this session`;
}

/* ── Run Evaluation ── */
async function runEvaluation() {
    const btn = document.getElementById('evalRunBtn');
    const empty = document.getElementById('evalEmpty');
    const avgSection = document.getElementById('evalAverages');
    const resultsSection = document.getElementById('evalResults');

    if (sessionQuestions.length === 0) {
        empty.textContent = 'No questions yet — ask something in the Chat tab first.';
        empty.style.display = 'block';
        avgSection.style.display = 'none';
        resultsSection.innerHTML = '';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Scoring...';
    empty.textContent = `Scoring ${sessionQuestions.length} question${sessionQuestions.length > 1 ? 's' : ''} in parallel…`;
    empty.style.display = 'block';
    avgSection.style.display = 'none';
    resultsSection.innerHTML = '';

    try {
        // Score one at a time with a gap to avoid Groq TPM rate limits.
        // llama-3.1-8b-instant has 20k TPM; ~1300 tokens/request → 4s gap keeps us under.
        // Retry once on 429 after a longer wait.
        const INTER_REQUEST_DELAY = 4000;
        const scoreResults = [];
        for (let i = 0; i < sessionQuestions.length; i++) {
            if (i > 0) await new Promise(r => setTimeout(r, INTER_REQUEST_DELAY));
            const q = sessionQuestions[i];
            empty.textContent = `Scoring question ${i + 1}/${sessionQuestions.length}…`;
            const ctxTexts = q.context_texts && q.context_texts.length > 0
                ? q.context_texts
                : (q.sources || []).map(s => s.content_preview).filter(Boolean);
            const payload = JSON.stringify({
                question: q.question,
                answer: q.answer || '',
                context_texts: ctxTexts,
                trace_id: q.trace_id || null,
            });
            const fetchScore = async () => {
                const res = await fetch('/api/eval/score', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload,
                });
                if (res.status === 429) {
                    // Rate limited — wait 6s and retry once
                    empty.textContent = `Rate limited, retrying question ${i + 1}…`;
                    await new Promise(r => setTimeout(r, 6000));
                    const retry = await fetch('/api/eval/score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: payload,
                    });
                    return retry.ok ? retry.json() : {};
                }
                return res.ok ? res.json() : {};
            };
            try {
                const data = await fetchScore();
                scoreResults.push({ question: q.question, answer: q.answer || '', scores: data.scores || {}, sources: q.sources || [] });
            } catch (e) {
                scoreResults.push({ question: q.question, answer: q.answer || '', scores: {}, sources: q.sources || [] });
            }
        }

        empty.style.display = 'none';

        // Compute averages
        const totals = { hallucination: [], answer_relevance: [], context_precision: [] };
        scoreResults.forEach(r => {
            Object.keys(totals).forEach(k => { if (r.scores[k] != null) totals[k].push(r.scores[k]); });
        });
        const avg = {};
        Object.keys(totals).forEach(k => {
            avg[k] = totals[k].length ? totals[k].reduce((a, b) => a + b, 0) / totals[k].length : null;
        });

        // Average score cards
        avgSection.style.display = 'block';
        document.getElementById('avgCards').innerHTML = ['hallucination', 'answer_relevance', 'context_precision'].map(metric => {
            const val = avg[metric];
            const pct = val != null ? Math.round(val * 100) : null;
            const label = { hallucination: 'Hallucination', answer_relevance: 'Answer Relevance', context_precision: 'Context Precision' }[metric];
            const color = metric === 'hallucination'
                ? (pct > 30 ? '#e53935' : pct > 15 ? '#fb8c00' : '#43a047')
                : (pct > 60 ? '#43a047' : pct > 30 ? '#fb8c00' : '#e53935');
            return `<div class="avg-card">
                <div class="avg-value" style="color:${color}">${pct !== null ? pct + '%' : 'N/A'}</div>
                <div class="avg-label">${label}</div>
                <div class="avg-bar"><div class="avg-bar-fill" style="width:${pct || 0}%;background:${color}"></div></div>
            </div>`;
        }).join('');

        const scoreColor = (val, invert) => {
            if (val === null) return '#999';
            return invert ? (val > 30 ? '#e53935' : val > 15 ? '#fb8c00' : '#43a047')
                          : (val > 60 ? '#43a047' : val > 30 ? '#fb8c00' : '#e53935');
        };

        // Per-question cards
        resultsSection.innerHTML = scoreResults.map((r, i) => {
            const s = r.scores || {};
            const hPct  = s.hallucination     != null ? Math.round(s.hallucination * 100)     : null;
            const arPct = s.answer_relevance  != null ? Math.round(s.answer_relevance * 100)  : null;
            const cpPct = s.context_precision != null ? Math.round(s.context_precision * 100) : null;
            const citationsHtml = r.sources && r.sources.length > 0 ? buildCiteRow(r.sources) : '';
            return `<div class="eval-card">
                <div class="eval-card-header" onclick="toggleEvalCard(${i})">
                    <div class="eval-card-left">
                        <span class="eval-card-num">${i + 1}</span>
                        <span class="eval-card-question">${escapeHtml(r.question)}</span>
                    </div>
                    <div class="eval-card-scores">
                        <span class="score-pill" style="background:${scoreColor(hPct,true)}">${hPct !== null ? hPct+'%' : '-'} H</span>
                        <span class="score-pill" style="background:${scoreColor(arPct,false)}">${arPct !== null ? arPct+'%' : '-'} R</span>
                        <span class="score-pill" style="background:${scoreColor(cpPct,false)}">${cpPct !== null ? cpPct+'%' : '-'} P</span>
                    </div>
                    <span class="eval-chevron" id="chevron-${i}">▼</span>
                </div>
                <div class="eval-card-body" id="evalBody-${i}" style="display:none">
                    <p>${escapeHtml(r.answer || '').replace(/\n/g,'<br>')}</p>
                    ${citationsHtml}
                </div>
            </div>`;
        }).join('');

    } catch (error) {
        empty.textContent = 'Error: ' + error.message;
        empty.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Evaluation';
    }
}

function toggleEvalCard(i) {
    const body = document.getElementById('evalBody-' + i);
    const chevron = document.getElementById('chevron-' + i);
    const open = body.style.display === 'none';
    body.style.display = open ? 'block' : 'none';
    chevron.textContent = open ? '▲' : '▼';
}

function buildCiteRow(sources) {
    // Group pages by document name
    const grouped = {};
    sources.forEach(src => {
        const name = src.document ? src.document.split(' — ')[1] || src.document : 'Unknown';
        if (!grouped[name]) grouped[name] = [];
        const page = src.page || '?';
        if (!grouped[name].includes(page)) grouped[name].push(page);
    });
    const chips = Object.entries(grouped).map(([name, pages]) =>
        `<span class="cite-tag">📄 ${escapeHtml(name)} · pg.${pages.join(', ')}</span>`
    ).join('');
    return `<div class="cite-row"><span class="cite-label">Sources:</span>${chips}</div>`;
}

function insertEvalCard(afterMsgId) {
    const messages = document.getElementById('messages');
    const afterEl = document.getElementById(afterMsgId);
    const id = 'eval-card-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'eval-inline-card';
    div.innerHTML = `<span class="eval-inline-label">📊 Evaluation</span><span class="eval-pending">Scoring…</span>`;
    if (afterEl && afterEl.nextSibling) {
        messages.insertBefore(div, afterEl.nextSibling);
    } else {
        messages.appendChild(div);
    }
    messages.scrollTop = messages.scrollHeight;
    return id;
}

async function runAutoEval(question, answer, contextTexts, traceId, reportId) {
    try {
        const res = await fetch('/api/eval/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer, context_texts: contextTexts, trace_id: traceId }),
        });
        if (!res.ok) {
            const el = document.getElementById(reportId);
            if (el) el.innerHTML = `<span class="eval-pending">Scoring unavailable</span>`;
            return;
        }
        const data = await res.json();
        if (data.skipped) { document.getElementById(reportId)?.remove(); return; }
        fillEvalCard(reportId, data.scores || {});
    } catch (e) {
        const el = document.getElementById(reportId);
        if (el) el.innerHTML = `<span class="eval-pending">Scoring unavailable</span>`;
    }
}

function fillEvalCard(reportId, scores) {
    const el = document.getElementById(reportId);
    if (!el) return;
    const hPct  = scores.hallucination       != null ? Math.round(scores.hallucination * 100)       : null;
    const arPct = scores.answer_relevance    != null ? Math.round(scores.answer_relevance * 100)    : null;
    const cpPct = scores.context_precision   != null ? Math.round(scores.context_precision * 100)   : null;
    const color = (val, invert) => {
        if (val === null) return '#aaa';
        return invert ? (val > 30 ? '#e53935' : val > 15 ? '#fb8c00' : '#43a047')
                      : (val > 60 ? '#43a047' : val > 30 ? '#fb8c00' : '#e53935');
    };
    const metric = (label, val, invert) =>
        `<span class="eval-metric-item">
            <span class="eval-metric-name">${label}</span>
            <span class="eval-metric-val" style="color:${color(val, invert)}">${val !== null ? val + '%' : '—'}</span>
        </span>`;
    el.innerHTML = `
        ${metric('H', hPct, true)}
        <span class="eval-metric-sep">·</span>
        ${metric('R', arPct, false)}
        <span class="eval-metric-sep">·</span>
        ${metric('P', cpPct, false)}
    `;
    document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}

function updateQuestionWithScores(msgId, scores) {
    // Called from runEvaluation() in eval tab — update existing eval card if present
    // Find the eval card that was inserted after the assistant message for this question
    fillEvalCard('eval-card-' + msgId, scores);
}

function toggleCitations(idOrIndex) {
    // Handle both string IDs (chat citations) and numeric indices (eval citations)
    if (typeof idOrIndex === 'string') {
        // Chat window citations - idOrIndex is the full element ID
        const content = document.getElementById(idOrIndex);
        const button = content ? content.previousElementSibling : null;
        if (content && button) {
            const isHidden = content.style.display === 'none';
            content.style.display = isHidden ? 'block' : 'none';
            button.innerHTML = button.innerHTML.replace(isHidden ? '▶' : '▼', isHidden ? '▼' : '▶');
        }
    } else {
        // Evaluation tab citations - idOrIndex is a numeric index
        const content = document.getElementById('citations-' + idOrIndex);
        const label = document.getElementById('citationsLabel-' + idOrIndex);
        if (content && label) {
            const open = content.style.display === 'none';
            content.style.display = open ? 'block' : 'none';
            label.textContent = open ? '▼ Hide citations' : `▶ Show ${content.children.length} citation${content.children.length > 1 ? 's' : ''}`;
        }
    }
}
