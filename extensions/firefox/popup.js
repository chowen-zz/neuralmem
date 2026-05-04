// popup.js - NeuralMem Popup UI Logic
// Handles quick save, search, and recent memories display.

const api = chrome || browser;

document.addEventListener('DOMContentLoaded', () => {
  const btnSave = document.getElementById('btn-save');
  const btnTweets = document.getElementById('btn-tweets');
  const searchInput = document.getElementById('search-input');
  const searchStatus = document.getElementById('search-status');
  const recentList = document.getElementById('recent-list');
  const recentStatus = document.getElementById('recent-status');

  // ─── Quick Save ───────────────────────────────────────────────────
  btnSave.addEventListener('click', async () => {
    btnSave.textContent = 'Saving...';
    btnSave.disabled = true;
    try {
      const response = await api.runtime.sendMessage({ action: 'quick_save' });
      if (response?.success) {
        btnSave.textContent = 'Saved!';
        setTimeout(() => { btnSave.textContent = 'Quick Save Page'; btnSave.disabled = false; }, 1500);
        loadRecent();
      } else {
        btnSave.textContent = 'Save Failed';
        setTimeout(() => { btnSave.textContent = 'Quick Save Page'; btnSave.disabled = false; }, 1500);
      }
    } catch (err) {
      btnSave.textContent = 'Error';
      setTimeout(() => { btnSave.textContent = 'Quick Save Page'; btnSave.disabled = false; }, 1500);
    }
  });

  // ─── Capture Tweets ───────────────────────────────────────────────
  btnTweets.addEventListener('click', async () => {
    const tabs = await api.tabs.query({ active: true, currentWindow: true });
    const tab = tabs[0];
    if (!tab?.id) return;
    btnTweets.textContent = 'Capturing...';
    btnTweets.disabled = true;
    try {
      const response = await api.tabs.sendMessage(tab.id, { action: 'capture_tweets' });
      if (response?.success) {
        btnTweets.textContent = `Captured ${response.count} tweets`;
      } else {
        btnTweets.textContent = 'No tweets found';
      }
    } catch (err) {
      btnTweets.textContent = 'Error';
    }
    setTimeout(() => { btnTweets.textContent = 'Capture Tweets'; btnTweets.disabled = false; }, 2000);
  });

  // ─── Search ─────────────────────────────────────────────────────
  let searchDebounce;
  searchInput.addEventListener('input', (e) => {
    clearTimeout(searchDebounce);
    const query = e.target.value.trim();
    if (!query) {
      searchStatus.textContent = '';
      loadRecent();
      return;
    }
    searchStatus.textContent = 'Searching...';
    searchDebounce = setTimeout(async () => {
      try {
        const response = await api.runtime.sendMessage({ action: 'search_memories', query });
        if (response?.success) {
          renderList(response.results || []);
          searchStatus.textContent = `${(response.results || []).length} results`;
        } else {
          searchStatus.textContent = 'Search failed';
        }
      } catch (err) {
        searchStatus.textContent = 'Error';
      }
    }, 300);
  });

  // ─── Recent Memories ─────────────────────────────────────────────
  async function loadRecent() {
    recentStatus.textContent = 'Loading...';
    try {
      const response = await api.runtime.sendMessage({ action: 'get_recent', limit: 10 });
      if (response?.success) {
        renderList(response.memories || []);
        recentStatus.textContent = `${(response.memories || []).length} memories`;
      } else {
        recentStatus.textContent = 'Failed to load';
      }
    } catch (err) {
      recentStatus.textContent = 'Error';
    }
  }

  function renderList(items) {
    recentList.innerHTML = '';
    if (items.length === 0) {
      recentList.innerHTML = '<div class="status">No memories yet</div>';
      return;
    }
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'memory-item';
      div.innerHTML = `
        <div class="title">${escapeHtml(item.title || 'Untitled')}</div>
        <div class="url">${escapeHtml(item.url || '')}</div>
      `;
      div.addEventListener('click', () => {
        if (item.url) api.tabs.create({ url: item.url });
      });
      recentList.appendChild(div);
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  loadRecent();
});
