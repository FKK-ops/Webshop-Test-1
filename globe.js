// Wireframe globe with surface dots, connection arcs, and Saturn-style rings.
// Lazy-initialised when the CTA section enters the viewport.

const canvas = document.getElementById('globeCanvas');
if (canvas) {
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const start = async () => {
    const THREE = await import('https://esm.sh/three@0.160.0');
    initGlobe(THREE, canvas, reduced);
  };

  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          io.disconnect();
          start();
          break;
        }
      }
    }, { rootMargin: '200px' });
    io.observe(canvas);
  } else {
    start();
  }
}

function initGlobe(THREE, canvas, reduced) {
  const host = canvas.parentElement;
  const size = () => ({ w: host.clientWidth, h: host.clientHeight });

  const scene = new THREE.Scene();
  const { w, h } = size();
  const camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 100);
  camera.position.set(0, 0.4, 5.2);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const COLOR_INK   = 0x1c2a2e;
  const COLOR_BLUE  = 0x4f87a8;
  const COLOR_TEAL  = 0x3fa9a3;
  const COLOR_SAGE  = 0x7aa089;

  const root = new THREE.Group();
  root.rotation.x = 0.22;
  scene.add(root);

  // --- Wireframe globe -----------------------------------------------------
  const sphereGeo = new THREE.IcosahedronGeometry(1, 4);
  const wireGeo = new THREE.WireframeGeometry(sphereGeo);
  const wireMat = new THREE.LineBasicMaterial({
    color: COLOR_BLUE, transparent: true, opacity: 0.22
  });
  const wire = new THREE.LineSegments(wireGeo, wireMat);
  root.add(wire);

  // Faint solid sphere behind wireframe for depth occlusion
  const innerMat = new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.04
  });
  root.add(new THREE.Mesh(new THREE.SphereGeometry(0.985, 48, 48), innerMat));

  // --- Surface dots (Fibonacci sphere) ------------------------------------
  const N = 360;
  const positions = new Float32Array(N * 3);
  const dotsVec = [];
  for (let i = 0; i < N; i++) {
    const phi = Math.acos(1 - (2 * (i + 0.5)) / N);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const x = Math.sin(phi) * Math.cos(theta);
    const y = Math.cos(phi);
    const z = Math.sin(phi) * Math.sin(theta);
    positions[i * 3]     = x * 1.001;
    positions[i * 3 + 1] = y * 1.001;
    positions[i * 3 + 2] = z * 1.001;
    dotsVec.push(new THREE.Vector3(x, y, z));
  }
  const dotsGeo = new THREE.BufferGeometry();
  dotsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const dotsMat = new THREE.PointsMaterial({
    color: COLOR_TEAL,
    size: 0.028,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.85
  });
  const dots = new THREE.Points(dotsGeo, dotsMat);
  root.add(dots);

  // --- Animated connection arcs -------------------------------------------
  const arcGroup = new THREE.Group();
  root.add(arcGroup);

  const ARC_COUNT = 9;
  const arcs = [];
  const arcSegments = 60;

  function rebuildArc(arc) {
    const a = dotsVec[Math.floor(Math.random() * dotsVec.length)];
    let b;
    do { b = dotsVec[Math.floor(Math.random() * dotsVec.length)]; }
    while (a.distanceTo(b) < 0.7);
    const mid = a.clone().add(b).multiplyScalar(0.5).normalize().multiplyScalar(1.45);
    const curve = new THREE.QuadraticBezierCurve3(a.clone(), mid, b.clone());
    const pts = curve.getPoints(arcSegments);
    const flat = new Float32Array(pts.length * 3);
    for (let i = 0; i < pts.length; i++) {
      flat[i*3]   = pts[i].x;
      flat[i*3+1] = pts[i].y;
      flat[i*3+2] = pts[i].z;
    }
    arc.geometry.setAttribute('position', new THREE.BufferAttribute(flat, 3));
    arc.geometry.setDrawRange(0, 0);
    arc.userData.t = 0;
    arc.userData.life = 90 + Math.random() * 80;
  }

  for (let i = 0; i < ARC_COUNT; i++) {
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.LineBasicMaterial({
      color: i % 2 === 0 ? COLOR_TEAL : COLOR_BLUE,
      transparent: true,
      opacity: 0.65
    });
    const line = new THREE.Line(geo, mat);
    rebuildArc(line);
    line.userData.t = Math.random() * line.userData.life;
    arcGroup.add(line);
    arcs.push(line);
  }

  // --- Saturn-style rings --------------------------------------------------
  const ringGroup = new THREE.Group();
  ringGroup.rotation.x = Math.PI * 0.42;
  ringGroup.rotation.z = -0.18;
  root.add(ringGroup);

  function makeRing(rIn, rOut, color, opacity) {
    const geo = new THREE.RingGeometry(rIn, rOut, 160);
    const mat = new THREE.MeshBasicMaterial({
      color, side: THREE.DoubleSide, transparent: true, opacity
    });
    return new THREE.Mesh(geo, mat);
  }
  ringGroup.add(makeRing(1.55, 1.58, COLOR_TEAL, 0.55));
  ringGroup.add(makeRing(1.68, 1.74, COLOR_BLUE, 0.35));
  ringGroup.add(makeRing(1.82, 1.84, COLOR_SAGE, 0.45));

  // Dotted outer ring (Points along a circle)
  const ringDotCount = 220;
  const ringDotPos = new Float32Array(ringDotCount * 3);
  for (let i = 0; i < ringDotCount; i++) {
    const a = (i / ringDotCount) * Math.PI * 2;
    const r = 1.95 + (Math.random() - 0.5) * 0.04;
    ringDotPos[i*3]   = Math.cos(a) * r;
    ringDotPos[i*3+1] = 0;
    ringDotPos[i*3+2] = Math.sin(a) * r;
  }
  const ringDotsGeo = new THREE.BufferGeometry();
  ringDotsGeo.setAttribute('position', new THREE.BufferAttribute(ringDotPos, 3));
  const ringDotsMat = new THREE.PointsMaterial({
    color: COLOR_INK, size: 0.018, sizeAttenuation: true, transparent: true, opacity: 0.55
  });
  ringGroup.add(new THREE.Points(ringDotsGeo, ringDotsMat));

  // --- Resize handling -----------------------------------------------------
  const ro = new ResizeObserver(() => {
    const { w, h } = size();
    if (w === 0 || h === 0) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  });
  ro.observe(host);

  // --- Animation loop ------------------------------------------------------
  let running = true;
  const visIo = new IntersectionObserver((entries) => {
    for (const e of entries) running = e.isIntersecting;
  }, { threshold: 0 });
  visIo.observe(canvas);

  function tick() {
    if (running) {
      if (!reduced) {
        root.rotation.y += 0.0028;
        ringGroup.rotation.y += 0.0014;
      }

      // animate arcs: draw progressively, hold, fade, restart
      for (const arc of arcs) {
        arc.userData.t += 1;
        const life = arc.userData.life;
        const p = arc.userData.t / life;
        if (p >= 1) {
          rebuildArc(arc);
          arc.material.opacity = 0.65;
          continue;
        }
        if (p < 0.55) {
          const draw = Math.floor((p / 0.55) * (arcSegments + 1));
          arc.geometry.setDrawRange(0, draw);
          arc.material.opacity = 0.75;
        } else {
          arc.geometry.setDrawRange(0, arcSegments + 1);
          arc.material.opacity = Math.max(0, 0.75 * (1 - (p - 0.55) / 0.45));
        }
      }

      renderer.render(scene, camera);
    }
    requestAnimationFrame(tick);
  }
  tick();
}
