/**
 * CRNN OCR Engine — app.js
 * All UI logic: drag-drop, file upload, API calls, results render
 */

'use strict';

// ─── DOM refs ──────────────────────────────────────────────
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const uploadCard    = document.getElementById('uploadCard');
const loadingOverlay= document.getElementById('loadingOverlay');
const resultsSection= document.getElementById('resultsSection');
const originalImg   = document.getElementById('originalImg');
const annotatedImg  = document.getElementById('annotatedImg');
const fullTextBox   = document.getElementById('fullTextBox');
const regionsGrid   = document.getElementById('regionsGrid');
const regionBadge   = document.getElementById('regionBadge');
const perfBadge     = document.getElementById('perfBadge');
const inferenceMs   = document.getElementById('inferenceMs');
const resultMeta    = document.getElementById('resultMeta');
const copyTextBtn   = document.getElementById('copyTextBtn');
const downloadBtn   = document.getElementById('downloadBtn');
const resetBtn      = document.getElementById('resetBtn');
const demoBtn       = document.getElementById('demoBtn');
const themeToggle   = document.getElementById('themeToggle');
const themeIcon     = document.getElementById('themeIcon');
const codeCopyBtn   = document.getElementById('codeCopyBtn');
const statRequests  = document.getElementById('statRequests');
const statChars     = document.getElementById('statChars');
const statAvgMs     = document.getElementById('statAvgMs');

let lastAnnotatedB64 = '';
let lastFullText     = '';

// ─── Theme toggle ─────────────────────────────────────────
const savedTheme = localStorage.getItem('ocr-theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next    = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('ocr-theme', next);
  updateThemeIcon(next);
});

function updateThemeIcon(theme) {
  themeIcon.className = theme === 'dark'
    ? 'bi bi-sun-fill'
    : 'bi bi-moon-fill';
}

// ─── Stats polling ────────────────────────────────────────
async function fetchStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    animateCount(statRequests, parseInt(statRequests.dataset.val || 0), data.total_requests);
    animateCount(statChars,    parseInt(statChars.dataset.val    || 0), data.total_chars_recognized);
    statRequests.dataset.val = data.total_requests;
    statChars.dataset.val    = data.total_chars_recognized;
    statAvgMs.textContent    = data.avg_inference_ms > 0
      ? data.avg_inference_ms + ' ms'
      : '—';
  } catch (_) {}
}

function animateCount(el, from, to) {
  if (from === to) return;
  const diff     = to - from;
  const duration = 800;
  const start    = performance.now();
  function step(now) {
    const t   = Math.min((now - start) / duration, 1);
    const val = Math.round(from + diff * easeOut(t));
    el.textContent = val.toLocaleString();
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

fetchStats();
setInterval(fetchStats, 5000);

// ─── Drag & Drop ──────────────────────────────────────────
['dragenter','dragover'].forEach(ev => {
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    uploadCard.classList.add('drag-active');
  });
});
['dragleave','drop'].forEach(ev => {
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    uploadCard.classList.remove('drag-active');
  });
});
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', e => {
  if (e.target.closest('label, button')) return;
  fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ─── Demo button ──────────────────────────────────────────
demoBtn.addEventListener('click', async () => {
  // Create a demo canvas image with text
  const canvas  = document.createElement('canvas');
  canvas.width  = 256;
  canvas.height = 64;
  const ctx     = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 256, 64);
  ctx.fillStyle = '#111111';
  ctx.font      = 'bold 42px serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const words   = ['hello', 'world', 'india', 'python', 'neural'];
  const word    = words[Math.floor(Math.random() * words.length)];
  ctx.fillText(word, 128, 32);

  canvas.toBlob(async blob => {
    const file = new File([blob], `demo_${word}.png`, { type: 'image/png' });
    handleFile(file);
  }, 'image/png');
});

// ─── Core: handle file → API → render ────────────────────
async function handleFile(file) {
  // Validate
  const allowed = ['image/png','image/jpeg','image/jpg',
                   'image/bmp','image/tiff','image/webp'];
  if (!allowed.includes(file.type)) {
    showToast('Unsupported file type. Use PNG, JPG, BMP, or WebP.', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Max 10 MB.', 'error');
    return;
  }

  // Show loading
  showLoading();
  animateLoadingSteps();

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/ocr', {
      method: 'POST',
      body  : formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hideLoading();
    renderResults(data);
    fetchStats();

  } catch (err) {
    hideLoading();
    showToast(`OCR failed: ${err.message}`, 'error');
    console.error(err);
  }
}

// ─── Render results ───────────────────────────────────────
function renderResults(data) {
  // Store for download/copy
  lastAnnotatedB64 = data.annotated_b64;
  lastFullText     = data.full_text || '—';

  // Images
  originalImg.src  = data.original_b64;
  annotatedImg.src = data.annotated_b64;

  // Text output
  fullTextBox.textContent = data.full_text || '(no text detected)';

  // Meta
  regionBadge.textContent = `${data.region_count} region${data.region_count !== 1 ? 's' : ''}`;
  inferenceMs.textContent  = data.inference_ms;
  resultMeta.textContent   =
    `${data.filename} · ${data.image_size.width}×${data.image_size.height}px · ` +
    `${data.region_count} region${data.region_count !== 1 ? 's' : ''} · ` +
    `${data.inference_ms} ms · ${data.timestamp.split('T')[0]}`;

  // Region chips
  regionsGrid.innerHTML = '';
  if (data.regions && data.regions.length > 0) {
    data.regions.forEach((r, i) => {
      const conf    = Math.round(r.confidence * 100);
      const chip    = document.createElement('div');
      chip.className = 'region-chip';
      chip.innerHTML = `
        <div class="region-chip-text">${escHtml(r.text)}</div>
        <div class="d-flex flex-column gap-1 align-items-end">
          <div class="region-chip-conf">${conf}%</div>
          <div class="conf-bar">
            <div class="conf-bar-fill" style="width:${conf}%"></div>
          </div>
        </div>
      `;
      regionsGrid.appendChild(chip);
    });
  } else {
    regionsGrid.innerHTML =
      '<span style="color:var(--text-muted);font-size:13px;padding:8px">No regions detected</span>';
  }

  // Show results section
  resultsSection.style.display = 'block';
  resultsSection.classList.add('fade-in');
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);

  showToast(
    `✓ Recognized "${data.full_text || 'text'}" in ${data.inference_ms} ms`,
    'success'
  );
}

// ─── Reset ────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  resultsSection.style.display = 'none';
  resultsSection.classList.remove('fade-in');
  fileInput.value = '';
  lastAnnotatedB64 = '';
  lastFullText     = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
  uploadCard.classList.remove('scanning');
});

// ─── Copy text ────────────────────────────────────────────
copyTextBtn.addEventListener('click', async () => {
  if (!lastFullText) return;
  try {
    await navigator.clipboard.writeText(lastFullText);
    showToast('Text copied to clipboard', 'success');
    copyTextBtn.innerHTML = '<i class="bi bi-clipboard-check me-1"></i>Copied!';
    setTimeout(() => {
      copyTextBtn.innerHTML = '<i class="bi bi-clipboard me-1"></i>Copy Text';
    }, 2000);
  } catch (_) {
    showToast('Copy failed — please copy manually', 'error');
  }
});

// ─── Download annotated image ─────────────────────────────
downloadBtn.addEventListener('click', () => {
  if (!lastAnnotatedB64) return;
  const a    = document.createElement('a');
  a.href     = lastAnnotatedB64;
  a.download = `crnn_ocr_result_${Date.now()}.png`;
  a.click();
  showToast('Downloading annotated image…', 'info');
});

// ─── Code tabs ────────────────────────────────────────────
document.querySelectorAll('.code-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.code-pre').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.target)?.classList.add('active');
  });
});

codeCopyBtn.addEventListener('click', () => {
  const active = document.querySelector('.code-pre.active');
  if (!active) return;
  navigator.clipboard.writeText(active.textContent.trim()).then(() => {
    codeCopyBtn.innerHTML = '<i class="bi bi-clipboard-check"></i>';
    showToast('Code copied', 'success');
    setTimeout(() => {
      codeCopyBtn.innerHTML = '<i class="bi bi-clipboard"></i>';
    }, 2000);
  });
});

// ─── Loading helpers ──────────────────────────────────────
function showLoading() {
  loadingOverlay.style.display = 'flex';
  uploadCard.classList.add('scanning');
  // Reset steps
  ['step1','step2','step3','step4'].forEach(id => {
    const el = document.getElementById(id);
    el.className = 'step-item';
    el.querySelector('.step-icon').className = 'bi bi-circle step-icon';
  });
}

function hideLoading() {
  loadingOverlay.style.display = 'none';
  uploadCard.classList.remove('scanning');
}

const STEP_DELAYS = [200, 700, 1300, 1800];
const STEP_IDS    = ['step1','step2','step3','step4'];

function animateLoadingSteps() {
  STEP_IDS.forEach((id, i) => {
    setTimeout(() => {
      // Mark previous as done
      if (i > 0) {
        const prev = document.getElementById(STEP_IDS[i-1]);
        prev.className = 'step-item done';
        prev.querySelector('.step-icon').className =
          'bi bi-check-circle-fill step-icon';
      }
      // Mark current as active
      const el = document.getElementById(id);
      el.className = 'step-item active';
      el.querySelector('.step-icon').className =
        'bi bi-arrow-repeat step-icon spin';
    }, STEP_DELAYS[i]);
  });
}

// ─── Toast ────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const wrap  = document.getElementById('toastWrap');
  const icons = { success: 'bi-check-circle-fill', error: 'bi-x-circle-fill', info: 'bi-info-circle-fill' };
  const colors= { success: 'var(--accent)', error: '#ff5555', info: 'var(--accent2)' };

  const item  = document.createElement('div');
  item.className = `toast-item ${type}`;
  item.innerHTML = `
    <i class="bi ${icons[type]}" style="color:${colors[type]};font-size:16px;flex-shrink:0"></i>
    <span>${escHtml(message)}</span>
  `;
  wrap.appendChild(item);

  setTimeout(() => {
    item.style.opacity    = '0';
    item.style.transform  = 'translateX(20px)';
    item.style.transition = '0.3s ease';
    setTimeout(() => item.remove(), 300);
  }, 3500);
}

// ─── Utility ──────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

// Add spin style dynamically
const spinStyle = document.createElement('style');
spinStyle.textContent = `
  .spin { animation: spinIcon 0.7s linear infinite; display:inline-block; }
  @keyframes spinIcon { to { transform: rotate(360deg); } }
`;
document.head.appendChild(spinStyle);

// ─── Navbar scroll shadow ─────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.getElementById('mainNav');
  nav.style.boxShadow = window.scrollY > 10
    ? '0 4px 24px rgba(0,0,0,0.3)'
    : 'none';
}, { passive: true });
