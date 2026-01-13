"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

import styles from "./GooeyNav.module.css";

export type GooeyNavItem = {
  label: string;
  href: string;
};

type Props = {
  items: GooeyNavItem[];
  animationTime?: number;
  particleCount?: number;
  particleDistances?: [number, number];
  particleR?: number;
  timeVariance?: number;
  colors?: number[];
  initialActiveIndex?: number;
  className?: string;
};

const noise = (n = 1) => n / 2 - Math.random() * n;

const getXY = (distance: number, pointIndex: number, totalPoints: number) => {
  const angle = ((360 + noise(8)) / totalPoints) * pointIndex * (Math.PI / 180);
  return [distance * Math.cos(angle), distance * Math.sin(angle)] as const;
};

export default function GooeyNav({
  items,
  animationTime = 600,
  particleCount = 15,
  particleDistances = [90, 10],
  particleR = 100,
  timeVariance = 300,
  colors = [1, 2, 3, 1, 2, 3, 1, 4],
  initialActiveIndex = 0,
  className,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const navRef = useRef<HTMLUListElement | null>(null);
  const filterRef = useRef<HTMLSpanElement | null>(null);
  const textRef = useRef<HTMLSpanElement | null>(null);

  const resolvedInitialIndex = useMemo(() => {
    const idx = items.findIndex((i) => i.href === pathname);
    return idx >= 0 ? idx : initialActiveIndex;
  }, [items, pathname, initialActiveIndex]);

  const [activeIndex, setActiveIndex] = useState(resolvedInitialIndex);

  const createParticle = (i: number, t: number, d: [number, number], r: number) => {
    const rotate = noise(r / 10);
    return {
      start: getXY(d[0], particleCount - i, particleCount),
      end: getXY(d[1] + noise(7), particleCount - i, particleCount),
      time: t,
      scale: 1 + noise(0.2),
      color: colors[Math.floor(Math.random() * colors.length)],
      rotate: rotate > 0 ? (rotate + r / 20) * 10 : (rotate - r / 20) * 10,
    };
  };

  const makeParticles = (element: HTMLElement) => {
    const d = particleDistances;
    const r = particleR;
    const bubbleTime = animationTime * 2 + timeVariance;
    element.style.setProperty("--time", `${bubbleTime}ms`);

    for (let i = 0; i < particleCount; i++) {
      const t = animationTime * 2 + noise(timeVariance * 2);
      const p = createParticle(i, t, d, r);
      element.classList.remove(styles.effectActive);

      setTimeout(() => {
        const particle = document.createElement("span");
        const point = document.createElement("span");
        particle.classList.add(styles.particle);
        particle.style.setProperty("--start-x", `${p.start[0]}px`);
        particle.style.setProperty("--start-y", `${p.start[1]}px`);
        particle.style.setProperty("--end-x", `${p.end[0]}px`);
        particle.style.setProperty("--end-y", `${p.end[1]}px`);
        particle.style.setProperty("--time", `${p.time}ms`);
        particle.style.setProperty("--scale", `${p.scale}`);
        particle.style.setProperty("--color", `var(--color-${p.color}, white)`);
        particle.style.setProperty("--rotate", `${p.rotate}deg`);

        point.classList.add(styles.point);
        particle.appendChild(point);
        element.appendChild(particle);

        requestAnimationFrame(() => {
          element.classList.add(styles.effectActive);
        });

        setTimeout(() => {
          try {
            element.removeChild(particle);
          } catch {
            // ignore
          }
        }, t);
      }, 30);
    }
  };

  const updateEffectPosition = (liEl: HTMLElement) => {
    if (!containerRef.current || !filterRef.current || !textRef.current) return;
    const containerRect = containerRef.current.getBoundingClientRect();
    const pos = liEl.getBoundingClientRect();

    const stylesObj = {
      left: `${pos.x - containerRect.x}px`,
      top: `${pos.y - containerRect.y}px`,
      width: `${pos.width}px`,
      height: `${pos.height}px`,
    };
    Object.assign(filterRef.current.style, stylesObj);
    Object.assign(textRef.current.style, stylesObj);
    textRef.current.innerText = liEl.innerText;
  };

  const handleClick = (index: number) => {
    const ul = navRef.current;
    if (!ul) return;
    const liEl = ul.querySelectorAll("li")[index] as HTMLLIElement | undefined;
    if (!liEl) return;

    if (activeIndex === index) {
      // still navigate (e.g. from a sub route) if needed
      router.push(items[index]?.href || "#");
      return;
    }

    setActiveIndex(index);
    updateEffectPosition(liEl);

    if (filterRef.current) {
      const particles = filterRef.current.querySelectorAll(`.${styles.particle}`);
      particles.forEach((p) => filterRef.current?.removeChild(p));
    }

    if (textRef.current) {
      textRef.current.classList.remove(styles.effectTextActive);
      // force reflow
      void textRef.current.offsetWidth;
      textRef.current.classList.add(styles.effectTextActive);
    }

    if (filterRef.current) {
      makeParticles(filterRef.current);
    }

    router.push(items[index]?.href || "#");
  };

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleClick(index);
    }
  };

  // Sync active index with current pathname (when route changes without clicking)
  useEffect(() => {
    const idx = items.findIndex((i) => i.href === pathname);
    if (idx >= 0 && idx !== activeIndex) setActiveIndex(idx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname, items]);

  // Position effect on mount + resize
  useEffect(() => {
    if (!navRef.current || !containerRef.current) return;
    const lis = navRef.current.querySelectorAll("li");
    const activeLi = lis[activeIndex] as HTMLLIElement | undefined;
    if (activeLi) {
      updateEffectPosition(activeLi);
      textRef.current?.classList.add(styles.effectTextActive);
    }

    const resizeObserver = new ResizeObserver(() => {
      const currentActiveLi = navRef.current?.querySelectorAll("li")[activeIndex] as
        | HTMLLIElement
        | undefined;
      if (currentActiveLi) updateEffectPosition(currentActiveLi);
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIndex]);

  return (
    <div
      className={[styles.container, className].filter(Boolean).join(" ")}
      ref={containerRef}
      aria-label="Gooey navigation"
    >
      <nav className={styles.nav}>
        <ul ref={navRef} className={styles.list}>
          {items.map((item, index) => (
            <li
              key={`${item.href}-${item.label}`}
              className={[styles.item, index === activeIndex ? styles.active : ""].filter(Boolean).join(" ")}
            >
              <button
                type="button"
                className={styles.link}
                onClick={() => handleClick(index)}
                onKeyDown={(e) => handleKeyDown(e, index)}
                aria-current={index === activeIndex ? "page" : undefined}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <span className={[styles.effect, styles.effectFilter].join(" ")} ref={filterRef} />
      <span className={[styles.effect, styles.effectText].join(" ")} ref={textRef} />
    </div>
  );
}


