/* Kronos web UI service worker.
 *
 * Installable shell only: HTML navigations and /static/ assets may be cached.
 * /api/* is never treated as a source of truth while offline. Bump CACHE_VERSION
 * when the shell or this file changes so clients pick up a new cache.
 */
'use strict';

const CACHE_VERSION = 'v1';
const CACHE_NAME = 'kronos-shell-' + CACHE_VERSION;

function appBase() {
  const path = self.location.pathname || '/';
  const slash = path.lastIndexOf('/');
  return slash <= 0 ? '/' : path.slice(0, slash + 1);
}

function joinBase(base, name) {
  if (!name) return base;
  if (name.charAt(0) === '/') name = name.slice(1);
  if (base.endsWith('/')) return base + name;
  return base + '/' + name;
}

function isApiPath(pathname) {
  return pathname.indexOf('/api/') !== -1;
}

function isStaticPath(pathname) {
  return pathname.indexOf('/static/') !== -1;
}

function jsonOffline() {
  return new Response(
    JSON.stringify({
      error: '当前处于离线状态，无法加载行情或排名数据',
      offline: true,
    }),
    {
      status: 503,
      headers: {'Content-Type': 'application/json; charset=utf-8'},
    }
  );
}

function htmlOffline() {
  return new Response(
    '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">' +
      '<meta name="viewport" content="width=device-width, initial-scale=1">' +
      '<meta name="theme-color" content="#0a0e14"><title>Kronos 离线</title>' +
      '<style>body{margin:0;min-height:100vh;display:grid;place-items:center;' +
      'background:#0a0e14;color:#e8ecf1;font-family:system-ui,sans-serif;padding:24px}' +
      'main{max-width:28rem;text-align:center}h1{font-size:1.25rem}p{color:#6b7a8d;line-height:1.55}' +
      'a{color:#3b82f6}</style></head><body><main><h1>Kronos</h1>' +
      '<p>当前处于离线状态。已安装的界面壳可用，行情、排名和预测数据需要联网后刷新。</p>' +
      '<p><a href="./">重试</a></p></main></body></html>',
    {
      status: 503,
      headers: {'Content-Type': 'text/html; charset=utf-8'},
    }
  );
}

function precacheUrls(base) {
  return [
    joinBase(base, 'login'),
    joinBase(base, 'offline'),
    joinBase(base, 'manifest.webmanifest'),
    joinBase(base, 'static/icon-192.png'),
    joinBase(base, 'static/icon-512.png'),
    joinBase(base, 'static/apple-touch-icon.png'),
    joinBase(base, 'static/favicon-32.png'),
    joinBase(base, 'static/logo-64.png'),
  ];
}

async function precache() {
  const cache = await caches.open(CACHE_NAME);
  const urls = precacheUrls(appBase());
  await Promise.all(urls.map(async (url) => {
    try {
      const response = await fetch(url, {credentials: 'same-origin', cache: 'reload'});
      if (response && response.ok) {
        await cache.put(url, response.clone());
      }
    } catch (_error) {
      // Public shell only; a missing asset must not fail install.
    }
  }));
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response && response.ok) {
    await cache.put(request, response.clone());
  }
  return response;
}

async function networkOnlyApi(request) {
  try {
    return await fetch(request);
  } catch (_error) {
    return jsonOffline();
  }
}

async function navigationResponse(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const fresh = await fetch(request);
    if (fresh && fresh.ok) {
      const type = fresh.headers.get('content-type') || '';
      if (type.indexOf('text/html') !== -1) {
        await cache.put(request, fresh.clone());
      }
    }
    return fresh;
  } catch (_error) {
    const cached = await cache.match(request);
    if (cached) return cached;
    const offline = await cache.match(joinBase(appBase(), 'offline'));
    if (offline) return offline;
    return htmlOffline();
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    await precache();
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) => key.indexOf('kronos-shell-') === 0 && key !== CACHE_NAME)
        .map((key) => caches.delete(key))
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (isApiPath(url.pathname)) {
    event.respondWith(networkOnlyApi(request));
    return;
  }

  if (isStaticPath(url.pathname)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  const accept = request.headers.get('accept') || '';
  if (request.mode === 'navigate' || accept.indexOf('text/html') !== -1) {
    event.respondWith(navigationResponse(request));
  }
});
