/* rose & ritual — JS */

document.addEventListener('DOMContentLoaded', () => {

  // ── Navbar scroll state ──
  const nav = document.getElementById('nav');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 40);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // ── Hero particles ──
  const particleContainer = document.getElementById('particles');
  if (particleContainer) {
    const count = 30;
    for (let i = 0; i < count; i++) {
      const p = document.createElement('div');
      p.className = 'particle';
      p.style.cssText = `
        left: ${Math.random() * 100}%;
        bottom: ${Math.random() * 20}%;
        --dur: ${6 + Math.random() * 8}s;
        --delay: ${Math.random() * 8}s;
        width: ${1 + Math.random() * 2}px;
        height: ${1 + Math.random() * 2}px;
        opacity: ${0.3 + Math.random() * 0.4};
      `;
      particleContainer.appendChild(p);
    }
  }

  // ── Mobile menu ──
  const menuToggle = document.getElementById('menuToggle');
  const menuClose  = document.getElementById('menuClose');
  const mobileMenu = document.getElementById('mobileMenu');
  const overlay    = document.getElementById('overlay');

  const openMenu = () => {
    mobileMenu.classList.add('open');
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  };
  const closeMenu = () => {
    mobileMenu.classList.remove('open');
    if (!document.getElementById('cartDrawer').classList.contains('open')) {
      overlay.classList.remove('visible');
      document.body.style.overflow = '';
    }
  };

  menuToggle?.addEventListener('click', openMenu);
  menuClose?.addEventListener('click', closeMenu);
  document.querySelectorAll('.menu-link').forEach(l => l.addEventListener('click', closeMenu));

  // ── Cart ──
  const cartToggle = document.getElementById('cartToggle');
  const cartClose  = document.getElementById('cartClose');
  const cartDrawer = document.getElementById('cartDrawer');
  const cartCount  = document.getElementById('cartCount');
  const cartItems  = document.getElementById('cartItems');
  const cartTotal  = document.getElementById('cartTotal');

  let cart = [];

  const products = {
    1: { name: 'Angel Knit Half-Zip',    price: 119, color: 'Grau Melange' },
    2: { name: 'Rose Sleeve Longsleeve', price: 79,  color: 'Blush Pink' },
    3: { name: 'Rose Sleeve Longsleeve', price: 79,  color: 'Royal Blue' },
  };

  const openCart = () => {
    cartDrawer.classList.add('open');
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  };
  const closeCart = () => {
    cartDrawer.classList.remove('open');
    if (!mobileMenu.classList.contains('open')) {
      overlay.classList.remove('visible');
      document.body.style.overflow = '';
    }
  };

  cartToggle?.addEventListener('click', openCart);
  cartClose?.addEventListener('click', closeCart);
  overlay?.addEventListener('click', () => { closeCart(); closeMenu(); });

  const renderCart = () => {
    const total = cart.reduce((s, i) => s + i.price, 0);
    cartTotal.textContent = `€${total.toFixed(2)}`;
    const n = cart.length;
    cartCount.textContent = n;
    cartCount.classList.toggle('visible', n > 0);

    if (!cart.length) {
      cartItems.innerHTML = '<p class="cart-empty">Dein Warenkorb ist leer.</p>';
      return;
    }
    cartItems.innerHTML = cart.map((item, idx) => `
      <div class="cart-item">
        <div class="cart-item__img"></div>
        <div class="cart-item__info">
          <div class="cart-item__name">${item.name}</div>
          <div class="cart-item__detail">${item.color}</div>
          <div class="cart-item__price">€${item.price}</div>
        </div>
        <button class="cart-item__remove" data-idx="${idx}" aria-label="Entfernen">✕</button>
      </div>
    `).join('');

    cartItems.querySelectorAll('.cart-item__remove').forEach(btn => {
      btn.addEventListener('click', () => {
        cart.splice(Number(btn.dataset.idx), 1);
        renderCart();
      });
    });
  };

  document.querySelectorAll('.product-card__quick-add').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = Number(btn.dataset.id);
      const p = products[id];
      if (p) {
        cart.push({ ...p });
        renderCart();
        openCart();
        btn.textContent = '✓ Hinzugefügt';
        setTimeout(() => btn.textContent = '+ Warenkorb', 1500);
      }
    });
  });

  renderCart();

  // ── Campaign Video ──
  const videoWrap = document.getElementById('videoWrap');
  const placeholder = document.getElementById('videoPlaceholder');
  const video = document.getElementById('campaignVideo');
  const playBtn = document.getElementById('playBtn');

  const playVideo = () => {
    if (video && video.getAttribute('src') || video?.querySelector('source')?.src) {
      placeholder.style.display = 'none';
      video.classList.add('playing');
      video.play().catch(() => {
        placeholder.style.display = 'flex';
        video.classList.remove('playing');
      });
    }
  };

  playBtn?.addEventListener('click', playVideo);
  playBtn?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); playVideo(); }
  });

  // ── Newsletter ──
  const form = document.getElementById('newsletterForm');
  const emailInput = document.getElementById('emailInput');
  const success = document.getElementById('newsletterSuccess');

  form?.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!emailInput.value || !emailInput.value.includes('@')) {
      emailInput.style.borderColor = 'red';
      return;
    }
    success.classList.add('visible');
    form.querySelector('.btn-primary').disabled = true;
    form.querySelector('.btn-primary').textContent = '✓';
    emailInput.value = '';
  });

  // ── Intersection Observer — fade in ──
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll(
    '.product-card, .values__item, .section-header, .about__text, .about__visual, .feature-banner__content, .lookbook-item, .newsletter__inner'
  ).forEach((el, i) => {
    el.classList.add('fade-in');
    el.style.transitionDelay = `${(i % 4) * 80}ms`;
    io.observe(el);
  });

  // ── Image fallback: show SVG placeholder if img fails to load ──
  document.querySelectorAll('.product-card__img--front').forEach(img => {
    if (img.tagName === 'IMG') {
      img.addEventListener('error', () => {
        img.style.display = 'none';
        const placeholder = img.nextElementSibling;
        if (placeholder) placeholder.style.display = 'block';
      });
      // If already broken (cached 404)
      if (img.complete && img.naturalWidth === 0) {
        img.dispatchEvent(new Event('error'));
      }
    }
  });

});
