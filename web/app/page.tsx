'use client';

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { ArrowRight, Bot } from 'lucide-react';
import GridScan from '@/components/GridScan';
import Header from '@/components/Header';
import ShinyText from '@/components/ShinyText';

export default function IntroPage() {
  const router = useRouter();

  const handleStartDemo = () => {
    router.push('/demo');
  };

  const handleGoMembers = () => {
    router.push('/member');
  };

  return (
    <main className="min-h-screen relative overflow-hidden bg-[#0a0a0c] text-white">
      {/* Background - GridScan */}
      <div className="absolute inset-0 z-0 opacity-90 pointer-events-none">
        <GridScan
          sensitivity={0.55}
          lineThickness={1.5}
          linesColor="#4a3d6a"
          gridScale={0.12}
          scanColor="#4ade80"
          scanOpacity={0.7}
          bloomIntensity={1.0}
          noiseIntensity={0.02}
          scanDuration={1.5}
          scanDelay={1.0}
        />
      </div>
      <div className="absolute inset-0 z-[1] bg-gradient-to-b from-[#0a0a0c]/40 via-transparent to-[#0a0a0c]/60 pointer-events-none" />

      {/* Content */}
      <div className="relative z-10">
        <Header />

        <div className="min-h-[calc(100vh-96px)] flex flex-col items-center justify-center px-6 text-center">
          
          {/* Main Title - Shiny & Big */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-6"
          >
            <h1 className="text-6xl md:text-8xl font-bold font-display tracking-tight leading-tight">
              <ShinyText 
                text="Kiosk Agent" 
                disabled={false} 
                speed={2}
                className="block" 
                color="#34d399"
                shineColor="#a7f3d0"
              />
            </h1>
          </motion.div>

          {/* Subtitle - Minimal */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-lg md:text-xl text-white/50 max-w-xl mx-auto font-light leading-relaxed mb-12"
          >
            AI-driven GUI automation. <br className="hidden md:block" />
            Control interfaces with natural language.
          </motion.p>

          {/* CTA Buttons - Modern Tech Style */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="flex flex-col sm:flex-row items-center gap-4"
          >
            <button
              onClick={handleStartDemo}
              className="group relative inline-flex items-center gap-2 px-8 py-4 bg-white text-black text-sm font-bold tracking-wide uppercase rounded-full hover:scale-105 transition-transform duration-300"
            >
              <span>Launch Demo</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>

            <button
              onClick={handleGoMembers}
              className="group inline-flex items-center gap-2 px-8 py-4 bg-white/5 border border-white/10 text-white/80 text-sm font-bold tracking-wide uppercase rounded-full hover:bg-white/10 hover:text-white transition-all duration-300"
            >
              <span>Team</span>
            </button>
          </motion.div>

          {/* Subtle Hint */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1 }}
            className="mt-20 flex items-center gap-2 text-xs text-white/40 uppercase tracking-widest font-mono"
          >
            <Bot className="w-3 h-3" />
            <span>Powered by Agent Studio</span>
          </motion.div>
        </div>
      </div>
    </main>
  );
}
