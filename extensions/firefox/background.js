// background.js - NeuralMem Service Worker
// Handles sync with NeuralMem API, bookmark capture, and queued retries.

const NEURALMEM_API_BASE = 'http://localhost:8000/api/v1';
const SYNC_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

// ─── Lifecycle ──────────────────────────────────────────────────────

browser.runtime.onInstalled.addListener((details) => {
  console.log('[NeuralMem] Extension installed/updated:', details.reason);
  browser.storage.local.set({ neuralmem_enabled: true, api_base: NEURALMEM_API_BASE });
});

browser.runtime.onStartup.addListener(() => {
  console.log('[NeuralMem] Browser started');
  startPeriodicSync();
});

// ─── Alarm-based periodic sync ──────────────────────────────────────

browser.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'neuralmem_sync') {
    flushQueue();
    syncBookmarks();
  }
});

function startPeriodicSync() {
  browser.alarms.create('neuralmem_sync', { periodInMinutes: 5 });
}

// ─── Message Listener ───────────────────────────────────────────────

browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'quick_save') {
    captureActiveTab().then(result => sendResponse(result));
    return true;
  }

  if (request.action === 'search_memories') {
    searchNeuralMem(request.query).then(result => sendResponse(result));
    return true;
  }

  if (request.action === 'get_recent') {
    getRecentMemories(request.limit || 10).then(result => sendResponse(result));
    return true;
  }

  if (request.action === 'flush_queue') {
    flushQueue().then(result => sendResponse(result));
    return true;
  }
});

// ─── Tab Capture ────────────────────────────────────────────────────

async function captureActiveTab() {
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!tab?.id) return { success: false, error: 'No active tab' };

  try {
    const response = await browser.tabs.sendMessage(tab.id, { action: 'capture_page' });
    return response || { success: false, error: 'No response from content script' };
  } catch (err) {
    // Content script may not be injected; inject dynamically
    await browser.tabs.executeScript(tab.id, { file: 'content_script.js' });
    const response = await browser.tabs.sendMessage(tab.id, { action: 'capture_page' });
    return response;
  }
}

// ─── Bookmark Sync ──────────────────────────────────────────────────

async function syncBookmarks() {
  const tree = await browser.bookmarks.getRecent(50);
  const bookmarks = tree.map(node => ({
    type: 'bookmark',
    url: node.url,
    title: node.title,
    bookmark_id: node.id,
    parent_id: node.parentId,
    date_added: new Date(node.dateAdded).toISOString(),
    captured_at: new Date().toISOString()
  }));

  const results = [];
  for (const payload of bookmarks) {
    try {
      const res = await fetch(`${NEURALMEM_API_BASE}/memories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      results.push({ ok: res.ok, url: payload.url });
    } catch (err) {
      results.push({ ok: false, url: payload.url, error: err.message });
    }
  }
  console.log('[NeuralMem] Bookmark sync complete:', results.filter(r => r.ok).length, 'ok');
  return results;
}

// ─── Queue Flush (retry failed captures) ────────────────────────────

async function flushQueue() {
  const { neuralmem_queue } = await browser.storage.local.get('neuralmem_queue');
  if (!Array.isArray(neuralmem_queue) || neuralmem_queue.length === 0) {
    return { flushed: 0 };
  }

  const remaining = [];
  let flushed = 0;

  for (const payload of neuralmem_queue) {
    try {
      const res = await fetch(`${NEURALMEM_API_BASE}/memories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        flushed++;
      } else {
        remaining.push(payload);
      }
    } catch (err) {
      remaining.push(payload);
    }
  }

  await browser.storage.local.set({ neuralmem_queue: remaining });
  console.log(`[NeuralMem] Flushed ${flushed}/${neuralmem_queue.length} queued items`);
  return { flushed, remaining: remaining.length };
}

// ─── Search / Recent ────────────────────────────────────────────────

async function searchNeuralMem(query) {
  try {
    const res = await fetch(`${NEURALMEM_API_BASE}/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return { success: true, results: await res.json() };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function getRecentMemories(limit = 10) {
  try {
    const res = await fetch(`${NEURALMEM_API_BASE}/memories?limit=${limit}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return { success: true, memories: await res.json() };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// ─── Context Menu ───────────────────────────────────────────────────

browser.runtime.onInstalled.addListener(() => {
  browser.contextMenus.create({
    id: 'neuralmem_save_page',
    title: 'Save to NeuralMem',
    contexts: ['page', 'link']
  });
});

browser.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'neuralmem_save_page') {
    captureActiveTab();
  }
});
