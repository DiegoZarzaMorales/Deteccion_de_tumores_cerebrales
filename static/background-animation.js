(function () {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const particles = [];
  const gridLines = [];
  const particleCount = 200;
  const connectionDistance = 120;
  const gridSpacing = 60;
  let width;
  let height;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    gridLines.length = 0;

    for (let y = height * 0.4; y < height + 100; y += gridSpacing) {
      gridLines.push({ y: y, type: 'h' });
    }
    for (let x = -200; x < width + 200; x += gridSpacing) {
      gridLines.push({ x: x, type: 'v' });
    }
  }

  function createParticle() {
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.8,
      vy: (Math.random() - 0.5) * 0.8,
      size: Math.random() * 2 + 1,
      brightness: Math.random() * 0.5 + 0.5,
      pulse: Math.random() * Math.PI * 2
    };
  }

  function drawGrid(time) {
    const vanishY = height * 0.35;
    const vanishX = width / 2;

    gridLines.filter(line => line.type === 'h').forEach((line, index) => {
      const progress = (line.y - vanishY) / (height - vanishY);
      const alpha = Math.max(0, Math.min(0.15, progress * 0.2));
      const glow = Math.sin(time * 0.001 + index * 0.3) * 0.05 + 0.1;
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0, 150, 255, ${alpha + glow})`;
      ctx.moveTo(0, line.y);
      ctx.lineTo(width, line.y);
      ctx.stroke();
    });

    gridLines.filter(line => line.type === 'v').forEach((line, index) => {
      const glow = Math.sin(time * 0.0015 + index * 0.2) * 0.03 + 0.08;
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0, 120, 255, ${glow})`;
      ctx.moveTo(vanishX, vanishY);
      ctx.lineTo(line.x, height + 50);
      ctx.stroke();
    });
  }

  function drawParticles() {
    particles.forEach(particle => {
      particle.x += particle.vx;
      particle.y += particle.vy;
      particle.pulse += 0.02;
      if (particle.x < 0) particle.x = width;
      if (particle.x > width) particle.x = 0;
      if (particle.y < 0) particle.y = height;
      if (particle.y > height) particle.y = 0;

      const pulse = Math.sin(particle.pulse) * 0.3 + 0.7;
      const alpha = particle.brightness * pulse;
      const gradient = ctx.createRadialGradient(
        particle.x, particle.y, 0,
        particle.x, particle.y, particle.size * 4
      );
      gradient.addColorStop(0, `rgba(100, 200, 255, ${alpha})`);
      gradient.addColorStop(0.5, `rgba(0, 150, 255, ${alpha * 0.3})`);
      gradient.addColorStop(1, 'rgba(0, 100, 200, 0)');
      ctx.beginPath();
      ctx.fillStyle = gradient;
      ctx.arc(particle.x, particle.y, particle.size * 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.fillStyle = `rgba(150, 220, 255, ${alpha})`;
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawConnections() {
    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < connectionDistance) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0, 180, 255, ${(1 - distance / connectionDistance) * 0.4})`;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function animate(time) {
    const background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, '#020817');
    background.addColorStop(0.5, '#061428');
    background.addColorStop(1, '#0a1e3a');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
    drawGrid(time);
    drawConnections();
    drawParticles();
    window.requestAnimationFrame(animate);
  }

  resize();
  for (let i = 0; i < particleCount; i += 1) {
    particles.push(createParticle());
  }
  window.addEventListener('resize', resize);
  window.requestAnimationFrame(animate);
}());
