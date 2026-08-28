import React, { useEffect, useRef, useState } from 'react';
import { Renderer, Program, Mesh, Triangle } from 'ogl';
import './MoltenMetal.css';

interface MoltenMetalProps {
  intensity?: 'high' | 'medium' | 'subtle';
  className?: string;
}

const VERT_SHADER = /* glsl */ `
  attribute vec2 position;
  attribute vec2 uv;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const FRAG_SHADER = /* glsl */ `
  precision highp float;

  uniform float uTime;
  uniform vec2 uResolution;
  uniform vec2 uMouse;
  uniform float uIntensity;
  varying vec2 vUv;

  // Simple pseudo 2D noise
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 4; ++i) {
      v += a * noise(p);
      p = rot * p * 2.0;
      a *= 0.5;
    }
    return v;
  }

  void main() {
    vec2 st = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
    vec2 mouse = (uMouse - 0.5 * uResolution.xy) / uResolution.y;

    float distToMouse = length(st - mouse);
    float mouseInfluence = smoothstep(0.6, 0.0, distToMouse);

    vec2 q = vec2(fbm(st * 2.5 + uTime * 0.08), fbm(st * 2.5 + vec2(5.2, 1.3) - uTime * 0.06));
    vec2 r = vec2(fbm(st * 2.5 + 4.0 * q + vec2(1.7, 9.2) + uTime * 0.12 + mouseInfluence * 0.3),
                  fbm(st * 2.5 + 4.0 * q + vec2(8.3, 2.8) - uTime * 0.1));

    float f = fbm(st * 2.5 + 4.0 * r);

    // Deep Dark Base with #5227FF (Primary Purple) and #FF9FFC (Accent Pink)
    vec3 colorBg = vec3(0.035, 0.039, 0.058); // #090A0F
    vec3 colorPrimary = vec3(0.32, 0.15, 1.0); // #5227FF
    vec3 colorAccent = vec3(1.0, 0.62, 0.98);  // #FF9FFC
    vec3 colorHighlight = vec3(0.9, 0.9, 1.0);

    vec3 col = mix(colorBg, colorPrimary * 0.45, clamp(f * f * 2.0, 0.0, 1.0));
    col = mix(col, colorAccent * 0.35, clamp(length(q), 0.0, 1.0));
    col = mix(col, colorHighlight * 0.6, clamp(pow(r.x, 3.0) * mouseInfluence, 0.0, 1.0));

    // Subtle grain
    float grain = (hash(gl_FragCoord.xy + uTime) - 0.5) * 0.03;
    col += grain;

    float alpha = uIntensity * smoothstep(0.0, 1.0, length(col) * 1.2);
    gl_FragColor = vec4(col, clamp(alpha, 0.0, uIntensity));
  }
`;

export const MoltenMetal: React.FC<MoltenMetalProps> = ({
  intensity = 'subtle',
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [webglFailed, setWebglFailed] = useState(false);

  const intensityValue = intensity === 'high' ? 0.75 : intensity === 'medium' ? 0.45 : 0.25;

  useEffect(() => {
    // Check reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setWebglFailed(true);
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    let renderer: Renderer | null = null;
    let animationFrameId: number;

    try {
      renderer = new Renderer({
        dpr: Math.min(window.devicePixelRatio, 2),
        alpha: true,
        premultipliedAlpha: true,
        antialias: true,
      });

      const gl = renderer.gl;
      if (!gl) {
        setWebglFailed(true);
        return;
      }

      container.appendChild(gl.canvas);
      gl.canvas.className = 'molten-metal-canvas';

      const geometry = new Triangle(gl);
      const program = new Program(gl, {
        vertex: VERT_SHADER,
        fragment: FRAG_SHADER,
        uniforms: {
          uTime: { value: 0 },
          uResolution: { value: [container.clientWidth, container.clientHeight] },
          uMouse: { value: [container.clientWidth * 0.5, container.clientHeight * 0.5] },
          uIntensity: { value: intensityValue },
        },
        transparent: true,
      });

      const mesh = new Mesh(gl, { geometry, program });

      const handleResize = () => {
        if (!container || !renderer) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        renderer.setSize(width, height);
        program.uniforms.uResolution.value = [width, height];
      };

      const handleMouseMove = (e: MouseEvent) => {
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = rect.height - (e.clientY - rect.top);
        program.uniforms.uMouse.value = [x, y];
      };

      window.addEventListener('resize', handleResize);
      window.addEventListener('mousemove', handleMouseMove);
      handleResize();

      let isPaused = false;
      const handleVisibilityChange = () => {
        isPaused = document.hidden;
      };
      document.addEventListener('visibilitychange', handleVisibilityChange);

      const render = (time: number) => {
        if (!isPaused && renderer) {
          program.uniforms.uTime.value = time * 0.001;
          renderer.render({ scene: mesh });
        }
        animationFrameId = requestAnimationFrame(render);
      };

      animationFrameId = requestAnimationFrame(render);

      return () => {
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener('resize', handleResize);
        window.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('visibilitychange', handleVisibilityChange);

        if (renderer && renderer.gl && renderer.gl.canvas) {
          if (renderer.gl.canvas.parentNode) {
            renderer.gl.canvas.parentNode.removeChild(renderer.gl.canvas);
          }
        }
      };
    } catch (err) {
      console.warn('MoltenMetal WebGL initialization failed. Falling back to static gradient:', err);
      setWebglFailed(true);
    }
  }, [intensityValue]);

  if (webglFailed) {
    return <div className={`molten-metal-fallback ${className}`} />;
  }

  return <div ref={containerRef} className={`molten-metal-container ${className}`} />;
};

export default MoltenMetal;
