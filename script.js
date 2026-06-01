(function () {
  'use strict';

  // ===== Sticky nav shadow on scroll
  const navShell = document.getElementById('navShell');
  if (navShell) {
    const onScroll = () => {
      if (window.scrollY > 8) navShell.classList.add('scrolled');
      else navShell.classList.remove('scrolled');
    };
    document.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // ===== Mobile menu
  const navToggle = document.getElementById('navToggle');
  const mobilePanel = document.getElementById('mobilePanel');
  if (navToggle && mobilePanel) {
    const closeMenu = () => {
      document.body.classList.remove('menu-open');
      navToggle.setAttribute('aria-expanded', 'false');
      mobilePanel.setAttribute('aria-hidden', 'true');
    };
    navToggle.addEventListener('click', () => {
      const open = document.body.classList.toggle('menu-open');
      navToggle.setAttribute('aria-expanded', String(open));
      mobilePanel.setAttribute('aria-hidden', String(!open));
    });
    mobilePanel.querySelectorAll('a').forEach((a) => {
      a.addEventListener('click', closeMenu);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && document.body.classList.contains('menu-open')) closeMenu();
    });
  }

  // ===== Highlight today in opening hours
  (function highlightToday() {
    const today = new Date().getDay(); // 0 Sun ... 6 Sat
    const rows = document.querySelectorAll('#hoursTable .hours-row');
    rows.forEach((r) => {
      if (parseInt(r.dataset.day, 10) === today) {
        r.classList.add('today');
        const badge = r.querySelector('.badge');
        if (badge) badge.hidden = false;
      }
    });
  })();

  // ===== Update live date in mockup
  (function liveDate() {
    const el = document.getElementById('mockDate');
    if (!el) return;
    const days = ['Sonntag', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag'];
    const months = [
      'Januar', 'Februar', 'März', 'April', 'Mai', 'Juni',
      'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember'
    ];
    const d = new Date();
    el.textContent = `${days[d.getDay()]} · ${d.getDate()}. ${months[d.getMonth()]} ${d.getFullYear()}`;
  })();

  // ===== Spline 3D hero scene (lazy)
  (function loadSpline() {
    const host = document.querySelector('.hero-3d');
    if (!host) return;
    const url = host.dataset.splineUrl;
    if (!url) return; // no URL set → keep existing CSS arcs only
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const loader = document.createElement('script');
    loader.type = 'module';
    loader.src = 'https://unpkg.com/@splinetool/viewer@1.9.35/build/spline-viewer.js';
    loader.onload = () => {
      const viewer = document.createElement('spline-viewer');
      viewer.setAttribute('url', url);
      viewer.setAttribute('loading-anim-type', 'none');
      viewer.setAttribute('events-target', 'global');
      host.appendChild(viewer);
      host.classList.add('is-loaded');
    };
    document.head.appendChild(loader);
  })();

  // ===== Praxis video — wire custom play overlay
  (function videoPlay() {
    const wrap = document.querySelector('.video-wrap');
    const video = document.getElementById('praxisVideo');
    const overlay = document.getElementById('videoPlayOverlay');
    if (!wrap || !video || !overlay) return;
    const start = () => {
      wrap.classList.add('is-playing');
      video.play().catch(() => {});
    };
    overlay.addEventListener('click', start);
    video.addEventListener('play', () => wrap.classList.add('is-playing'));
    video.addEventListener('pause', () => wrap.classList.remove('is-playing'));
    video.addEventListener('ended', () => wrap.classList.remove('is-playing'));
  })();

  // ===== Reveal on scroll (IntersectionObserver)
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        });
      },
      { rootMargin: '0px 0px -60px 0px', threshold: 0.05 }
    );
    document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));
  }
})();
