// content_script.js - NeuralMem Capture Logic
// Injected into all pages to capture webpage content, tweets, and bookmarks.

(function () {
  'use strict';

  const NEURALMEM_API_BASE = 'http://localhost:8000/api/v1';

  // ─── Utils ──────────────────────────────────────────────────────────

  function getMetaContent(name) {
    const el = document.querySelector(`meta[name="${name}"], meta[property="og:${name}"]`);
    return el ? el.getAttribute('content') : '';
  }

  function extractMainText() {
    // Simple heuristic: prefer article/main, fallback to body textContent
    const article = document.querySelector('article, main, [role="main"]');
    if (article) return article.innerText.trim();
    return document.body ? document.body.innerText.trim().slice(0, 8000) : '';
  }

  function extractTweets() {
    const tweets = [];
    const tweetEls = document.querySelectorAll('[data-testid="tweet"], article[data-testid="tweet"]');
    tweetEls.forEach((el, idx) => {
      const textEl = el.querySelector('[data-testid="tweetText"]');
      const timeEl = el.querySelector('time');
      const authorEl = el.querySelector('a[role="link"]');
      if (textEl) {
        tweets.push({
          index: idx,
          text: textEl.innerText.trim(),
          author: authorEl ? authorEl.getAttribute('href') : '',
          timestamp: timeEl ? timeEl.getAttribute('datetime') : null,
          url: window.location.href
        });
      }
    });
    return tweets;
  }

  function buildPagePayload() {
    return {
      type: 'page',
      url: window.location.href,
      title: document.title,
      description: getMetaContent('description'),
      content: extractMainText(),
      captured_at: new Date().toISOString()
    };
  }

  function buildTweetPayload(tweet) {
    return {
      type: 'tweet',
      ...tweet,
      captured_at: new Date().toISOString()
    };
  }

  // ─── API Sync ───────────────────────────────────────────────────────

  async function sendToNeuralMem(payload) {
    try {
      const response = await fetch(`${NEURALMEM_API_BASE}/memories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (err) {
      console.error('[NeuralMem] sync failed:', err);
      // Fallback: queue in localStorage for background retry
      queuePayload(payload);
      return null;
    }
  }

  function queuePayload(payload) {
    const queue = JSON.parse(localStorage.getItem('neuralmem_queue') || '[]');
    queue.push(payload);
    localStorage.setItem('neuralmem_queue', JSON.stringify(queue));
  }

  // ─── Message Listener (from popup / background) ─────────────────────

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'capture_page') {
      const payload = buildPagePayload();
      sendToNeuralMem(payload).then(result => {
        sendResponse({ success: !!result, payload });
      });
      return true; // async
    }

    if (request.action === 'capture_tweets') {
      const tweets = extractTweets();
      const payloads = tweets.map(buildTweetPayload);
      Promise.all(payloads.map(sendToNeuralMem)).then(results => {
        sendResponse({ success: true, count: tweets.length, payloads });
      });
      return true;
    }

    if (request.action === 'get_page_info') {
      sendResponse({
        url: window.location.href,
        title: document.title,
        description: getMetaContent('description'),
        tweetCount: extractTweets().length
      });
      return false;
    }
  });

  // ─── Auto-capture on significant pages (optional) ──────────────────

  if (window.location.hostname.includes('twitter.com') ||
      window.location.hostname.includes('x.com')) {
    // Observe new tweets on scroll
    const observer = new MutationObserver(() => {
      // Debounced auto-capture could go here
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
})();
