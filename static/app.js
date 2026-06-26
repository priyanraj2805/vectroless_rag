let documents = [];
let pollInterval = null;

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
    if (!pdfs.length) { addMessage('assistant', 'Please upload PDF files only.'); return; }

    const formData = new FormData();
    pdfs.forEach(f => formData.append('files', f));

    addMessage('assistant', `Uploading ${pdfs.length} file(s)...`);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        const names = data.documents.map(d => d.filename).join(', ');
        addMessage('assistant', `Processing: ${names}. This may take a minute while entities are extracted.`);
        loadDocuments();
        startPolling();
    } catch (error) {
        addMessage('assistant', 'Upload failed: ' + error.message);
    }
}

function startPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
        await loadDocuments();
        const hasProcessing = documents.some(d => d.status === 'processing');
        if (!hasProcessing) {
            clearInterval(pollInterval);
            pollInterval = null;
            loadStats();
        }
    }, 3000);
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        documents = data.documents;
        renderDocuments();
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

function renderDocuments() {
    const list = document.getElementById('fileList');
    if (!documents.length) {
        list.innerHTML = '<div class="empty-docs">No documents uploaded yet</div>';
        return;
    }
    list.innerHTML = documents.map(doc => `
        <div class="file-item ${doc.status}">
            <span class="name" title="${doc.filename}">${doc.filename}</span>
            <span class="status">${doc.status}</span>
            <span class="delete" onclick="deleteDocument(${doc.id}, event)" title="Delete">&times;</span>
        </div>
    `).join('');
}

async function deleteDocument(id, e) {
    e.stopPropagation();
    try {
        await fetch(`/api/documents/${id}`, { method: 'DELETE' });
        loadDocuments();
        loadStats();
    } catch (error) {
        console.error('Failed to delete:', error);
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question) return;

    addMessage('user', question);
    input.value = '';

    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;

    const thinkingId = addMessage('assistant', '<span class="thinking">Searching knowledge graph...</span>', true);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await response.json();

        let html = escapeHtml(data.answer).replace(/\n/g, '<br>');
        if (data.sources && data.sources.length > 0) {
            const refs = data.sources
                .filter(s => s.section || s.page)
                .map(s => `p.${s.page || '?'}${s.section ? ' — ' + escapeHtml(s.section) : ''}`)
                .join(' | ');
            if (refs) html += `<div class="sources-list"><span>Sources: ${refs}</span></div>`;
        }
        if (data.entities_found || data.chunks_used) {
            html += `<div class="sources-list"><span>${data.entities_found} entities · ${data.chunks_used} chunks used</span></div>`;
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
    const id = 'msg-' + Date.now();
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
            `Graph: ${data.nodes} nodes · ${data.edges} edges · ${data.chunks} chunks · ${data.documents} docs`;
    } catch (error) {
        document.getElementById('stats').textContent = 'Graph: not loaded';
    }
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
