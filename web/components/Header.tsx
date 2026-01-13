'use client';

import { motion } from 'framer-motion';
import { Github, ExternalLink } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

export default function Header() {
  return (
    <header className="sticky top-0 z-50 px-6 py-6 bg-[#0f0f12]/80 backdrop-blur-xl border-b border-white/5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Left: Branding */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-6"
        >
          {/* Logo → Home */}
          <Link href="/" aria-label="홈으로" className="flex items-center gap-6 group cursor-pointer">
            <div className="relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-violet-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-700" />
              <div className="relative w-16 h-16 rounded-2xl bg-[#1a1b1e] border border-white/10 flex items-center justify-center overflow-hidden">
                <Image 
                  src="/images/logo.png" 
                  alt="Pseudo Lab Logo" 
                  width={64} 
                  height={64}
                  className="w-full h-full object-cover p-1"
                />
              </div>
            </div>
            
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-bold tracking-tight text-white font-display group-hover:opacity-95 transition-opacity">
                  Agent Studio
                </h1>
                <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold tracking-wider uppercase">
                  Beta
                </span>
              </div>
              <p className="text-[#9aa0a6] text-sm font-medium flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                Pseudo Lab <span className="text-white/20">|</span> Agent Studio
              </p>
            </div>
          </Link>
        </motion.div>

        {/* Right: Team & Links */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-6"
        >
          <div className="text-right hidden sm:block">
            <p className="text-white font-medium text-sm leading-tight mb-1">
              김승혁 · 김재현 · 이규민 · 전민정
            </p>
            <p className="text-[#9aa0a6] text-[10px] uppercase tracking-widest font-bold">
              Made by Agent Studio
            </p>
          </div>
          
          <div className="h-8 w-px bg-white/10 hidden sm:block" />
          
          <div className="flex items-center gap-3">
            <a 
              href="https://github.com/ag-ui-protocol/ag-ui" 
              target="_blank" 
              className="p-2.5 rounded-xl bg-white/5 border border-white/5 text-[#9aa0a6] hover:bg-white/10 hover:text-white hover:border-white/20 transition-all duration-300 group"
            >
              <Github className="w-5 h-5 group-hover:scale-110 transition-transform" />
            </a>
            <a 
              href="https://gg.pseudo-lab.com/" 
              target="_blank" 
              className="p-2.5 rounded-xl bg-white/5 border border-white/5 text-[#9aa0a6] hover:bg-white/10 hover:text-white hover:border-white/20 transition-all duration-300 group"
            >
              <ExternalLink className="w-5 h-5 group-hover:scale-110 transition-transform" />
            </a>
          </div>
        </motion.div>
      </div>
    </header>
  );
}
