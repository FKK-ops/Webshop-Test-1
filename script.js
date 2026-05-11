/* rose & ritual — minimal interactions */

document.addEventListener('DOMContentLoaded', () => {

  // ── Mobile menu ──
  const menuToggle = document.getElementById('menuToggle');
  const menuClose  = document.getElementById('menuClose');
  const mobileMenu = document.getElementById('mobileMenu');
  const overlay    = document.getElementById('overlay');
  const cartDrawer = document.getElementById('cartDrawer');

  const openMenu = () => {
    mobileMenu.classList.add('open');
    document.body.style.overflow = 'hidden';
  };
  const closeMenu = () => {
    mobileMenu.classList.remove('open');
    if (!cartDrawer.classList.contains('open')) document.body.style.overflow = '';
  };
  menuToggle?.addEventListener('click', openMenu);
  menuClose?.addEventListener('click', closeMenu);
  document.querySelectorAll('.menu-link').forEach(l => l.addEventListener('click', closeMenu));

  // ── Cart ──
  const cartToggle = document.getElementById('cartToggle');
  const cartClose  = document.getElementById('cartClose');
  const cartCount  = document.getElementById('cartCount');
  const cartItems  = document.getElementById('cartItems');
  const cartTotal  = document.getElementById('cartTotal');

  const products = {
    1: { name: 'Half-Zip Knit',         price: 119, sub: 'Heather Grey' },
    2: { name: 'Contrast Longsleeve',   price: 79,  sub: 'White / Rose' },
    3: { name: 'Contrast Longsleeve',   price: 79,  sub: 'White / Cobalt' },
  };

  let cart = [];

  const openCart = () => {
    cartDrawer.classList.add('open');
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  };
  const closeCart = () => {
    cartDrawer.classList.remove('open');
    overlay.classList.remove('visible');
    if (!mobileMenu.classList.contains('open')) document.body.style.overflow = '';
  };

  cartToggle?.addEventListener('click', openCart);
  cartClose?.addEventListener('click', closeCart);
  overlay?.addEventListener('click', closeCart);

  const renderCart = () => {
    cartCount.textContent = `(${cart.length})`;
    const total = cart.reduce((s, i) => s + i.price, 0);
    cartTotal.textContent = `€${total}`;

    if (!cart.length) {
      cartItems.innerHTML = '<p class="cart__empty">Your bag is empty.</p>';
      return;
    }

    cartItems.innerHTML = cart.map((item, idx) => `
      <div class="cart-item">
        <div class="cart-item__img"></div>
        <div class="cart-item__info">
          <span class="cart-item__name">${item.name}</span>
          <span class="cart-item__sub">${item.sub}</span>
          <span class="cart-item__price">€${item.price}</span>
        </div>
        <button class="cart-item__rm" data-idx="${idx}">Remove</button>
      </div>
    `).join('');

    cartItems.querySelectorAll('.cart-item__rm').forEach(btn => {
      btn.addEventListener('click', () => {
        cart.splice(Number(btn.dataset.idx), 1);
        renderCart();
      });
    });
  };

  document.querySelectorAll('.card__add').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const p = products[Number(btn.dataset.id)];
      if (!p) return;
      cart.push({ ...p });
      renderCart();
      openCart();
      const orig = btn.textContent;
      btn.textContent = 'Added ✓';
      setTimeout(() => btn.textContent = orig, 1400);
    });
  });

  renderCart();

  // ── Product filter ──
  const filters = document.querySelectorAll('.filter');
  const cards = document.querySelectorAll('#productGrid .card');
  filters.forEach(f => {
    f.addEventListener('click', () => {
      filters.forEach(x => x.classList.remove('active'));
      f.classList.add('active');
      const cat = f.dataset.filter;
      cards.forEach(c => {
        c.classList.toggle('hidden', cat !== 'all' && c.dataset.cat !== cat);
      });
    });
  });

  // ── Campaign video ──
  const video = document.getElementById('campaignVideo');
  const playBtn = document.getElementById('playBtn');
  const videoWrap = document.getElementById('videoWrap');

  const playVideo = () => {
    if (!video) return;
    playBtn.style.display = 'none';
    video.classList.add('playing');
    video.play().catch(() => {
      playBtn.style.display = 'flex';
      video.classList.remove('playing');
    });
  };
  playBtn?.addEventListener('click', playVideo);
  videoWrap?.addEventListener('click', (e) => {
    if (e.target === videoWrap) playVideo();
  });

  // ── Newsletter ──
  const form = document.getElementById('newsletterForm');
  const emailInput = document.getElementById('emailInput');
  const success = document.getElementById('newsletterSuccess');

  form?.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!emailInput.value.includes('@')) {
      emailInput.style.borderColor = '#ff5555';
      return;
    }
    success.classList.add('visible');
    emailInput.value = '';
    form.querySelector('.btn').textContent = '✓ Done';
  });

  // ── Fade-in on scroll ──
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll(
    '.card, .look, .spec, .intro, .editorial__left, .editorial__right, .about__col, .news__inner, .lookbook__head, .shop__head'
  ).forEach((el, i) => {
    el.classList.add('fade');
    el.style.transitionDelay = `${Math.min((i % 6) * 60, 360)}ms`;
    io.observe(el);
  });

  // ── Show placeholders if images fail ──
  document.querySelectorAll('.card__img img').forEach(img => {
    if (img.complete && img.naturalWidth === 0) {
      img.style.display = 'none';
      const ph = img.nextElementSibling;
      if (ph) ph.style.display = 'flex';
    }
  });

  // Editorial & lookbook image fallbacks
  document.querySelectorAll('.editorial__img img, .look__img img').forEach(img => {
    const markPh = () => img.parentElement.classList.add(
      img.parentElement.classList.contains('editorial__img') ? 'editorial__img--ph' : 'look__img--ph'
    );
    img.addEventListener('error', markPh);
    if (img.complete && img.naturalWidth === 0) markPh();
  });

});
