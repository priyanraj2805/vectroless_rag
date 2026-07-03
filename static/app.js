let documents = [];
let pollInterval = null;
let liveTimerInterval = null;
let prevStatuses = {};
let selectedDocumentIds = [];

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
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !e.shiftKey) sendMessage(); });

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

    if (selectedDocumentIds.length > 0) {
        html += `<div class="all-docs-btn" onclick="clearDocumentSelection()">✕ Clear selection (${selectedDocumentIds.length})</div>`;
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

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question) return;

    addMessage('user', escapeHtml(question));
    input.value = '';

    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;

    const thinkingId = addMessage('assistant', '<span class="thinking">Thinking...</span>', true);

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

        if (data.sources && data.sources.length > 0) {
            const refs = data.sources
                .filter(s => s.section || s.page)
                .map(s => `p.${s.page || '?'}${s.section ? ' — ' + escapeHtml(s.section) : ''}`)
                .join(' · ');
            if (refs) html += `<div class="msg-meta">Sources: ${refs}</div>`;
        }

        if (data.chunks_used > 0) {
            html += `<div class="msg-meta">${data.entities_found} entities · ${data.chunks_used} chunks used</div>`;
        }

        updateMessage(thinkingId, html);
    } catch (error) {
        updateMessage(thinkingId, 'Error: ' + error.message);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

function addMessage(role, content, returnId = false) {
    const messages = document.getElementById('messages');
    const id = 'msg-' + Date.now() + Math.random();
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.id = id;
    div.innerHTML = `<div class="bubble">${content}</div>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return returnId ? id : undefined;
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
