/* rose & ritual — product detail page logic */

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

  let cart = JSON.parse(sessionStorage.getItem('rr_cart') || '[]');

  const saveCart = () => sessionStorage.setItem('rr_cart', JSON.stringify(cart));

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
          <span class="cart-item__sub">${item.sub}${item.size ? ' · ' + item.size : ''}</span>
          <span class="cart-item__price">€${item.price}</span>
        </div>
        <button class="cart-item__rm" data-idx="${idx}">Remove</button>
      </div>
    `).join('');

    cartItems.querySelectorAll('.cart-item__rm').forEach(btn => {
      btn.addEventListener('click', () => {
        cart.splice(Number(btn.dataset.idx), 1);
        saveCart();
        renderCart();
      });
    });
  };

  renderCart();

  // ── Gallery ──
  const mainImg = document.getElementById('pdpMainImg');
  const thumbs  = document.querySelectorAll('.pdp__thumb');

  thumbs.forEach((thumb, i) => {
    thumb.addEventListener('click', () => {
      thumbs.forEach(t => t.classList.remove('active'));
      thumb.classList.add('active');
      if (mainImg && thumb.dataset.src) {
        mainImg.src = thumb.dataset.src;
        mainImg.onerror = () => {
          mainImg.style.display = 'none';
          const ph = document.getElementById('pdpMainPh');
          if (ph) ph.style.display = 'flex';
        };
      }
    });
  });

  // ── Size selector ──
  const sizeButtons = document.querySelectorAll('.pdp__size:not(.oos)');
  const sizeNotice  = document.getElementById('sizeNotice');
  let selectedSize  = null;

  sizeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      sizeButtons.forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      selectedSize = btn.dataset.size;
      if (sizeNotice) sizeNotice.classList.remove('visible');
    });
  });

  // ── Add to Cart ──
  const addBtn     = document.getElementById('pdpAddBtn');
  const toast      = document.getElementById('pdpToast');
  const productData = window.PRODUCT_DATA;

  addBtn?.addEventListener('click', () => {
    if (!selectedSize) {
      if (sizeNotice) sizeNotice.classList.add('visible');
      return;
    }
    cart.push({ ...productData, size: selectedSize });
    saveCart();
    renderCart();
    openCart();
    showToast(`Added to bag — ${selectedSize}`);
  });

  const showToast = (msg) => {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2200);
  };

  // ── Accordion ──
  document.querySelectorAll('.pdp__acc-trigger').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const item = trigger.closest('.pdp__acc-item');
      const isOpen = item.classList.contains('open');
      document.querySelectorAll('.pdp__acc-item.open').forEach(i => i.classList.remove('open'));
      if (!isOpen) item.classList.add('open');
    });
  });

  // Open first accordion by default
  document.querySelector('.pdp__acc-item')?.classList.add('open');

  // ── Fade-in ──
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.card, .related').forEach((el, i) => {
    el.classList.add('fade');
    el.style.transitionDelay = `${Math.min((i % 4) * 60, 240)}ms`;
    io.observe(el);
  });

});
