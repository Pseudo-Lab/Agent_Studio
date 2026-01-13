'use client';

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { ArrowLeft, Github, Users } from 'lucide-react';
import GridScan from '@/components/GridScan';
import Header from '@/components/Header';
import ProfileCard from '@/components/ProfileCard';

interface TeamMember {
  id: string;
  name: string;
  nameKr: string;
  title: string;
  handle: string;
  status: string;
  avatarUrl: string;       // 중앙 큰 이미지 (LinkedIn QR)
  miniAvatarUrl: string;   // 하단 작은 동그라미 (GitHub 프로필)
  githubUrl?: string;
  linkedinUrl?: string;
  glowColor: string;
  innerGradient: string;
}

const TEAM_MEMBERS: TeamMember[] = [
  {
    id: 'seunghyeok',
    name: 'Seunghyeok Kim',
    nameKr: '김승혁',
    title: 'namu',
    handle: 'SeungHyeokKim',
    status: 'Agent Studio 11기 러너',
    avatarUrl: '/images/linkedin/seunghyuk.jpg',
    miniAvatarUrl: 'https://github.com/SeungHyeokKim.png',
    githubUrl: 'https://github.com/SeungHyeokKim',
    linkedinUrl: 'https://www.linkedin.com/in/승혁-김-9092b5306',
    glowColor: 'rgba(52, 211, 153, 0.6)',
    innerGradient: 'linear-gradient(145deg,#34d39933 0%,#6ee7b744 100%)',
  },
  {
    id: 'jaehyun',
    name: 'Jaehyun Kim',
    nameKr: '김재현',
    title: 'KTDS',
    handle: 'jh941213',
    status: 'Agent Studio 11기 빌더',
    avatarUrl: '/images/linkedin/jaehyun.jpg',
    miniAvatarUrl: 'https://github.com/jh941213.png',
    githubUrl: 'https://github.com/jh941213',
    linkedinUrl: 'https://www.linkedin.com/in/kjh941213/',
    glowColor: 'rgba(59, 130, 246, 0.6)',
    innerGradient: 'linear-gradient(145deg,#3b82f633 0%,#60a5fa44 100%)',
  },
  {
    id: 'kyumin',
    name: 'Kyumin Lee',
    nameKr: '이규민',
    title: 'KT',
    handle: 'qmin2',
    status: 'Agent Studio 11기 러너',
    avatarUrl: '/images/linkedin/lee.jpg',
    miniAvatarUrl: 'https://github.com/qmin2.png',
    githubUrl: 'https://github.com/qmin2',
    linkedinUrl: 'https://www.linkedin.com/in/kyumin-lee-620b43376/',
    glowColor: 'rgba(245, 158, 11, 0.6)',
    innerGradient: 'linear-gradient(145deg,#f59e0b33 0%,#fbbf2444 100%)',
  },
  {
    id: 'minjeong',
    name: 'Minjeong Jeon',
    nameKr: '전민정',
    title: 'AICESS',
    handle: 'ummjevel',
    status: 'Agent Studio 11기 러너',
    avatarUrl: '/images/linkedin/jeon.jpg',
    miniAvatarUrl: 'https://github.com/ummjevel.png',
    githubUrl: 'https://github.com/ummjevel',
    linkedinUrl: 'https://www.linkedin.com/in/mseagle2023/',
    glowColor: 'rgba(236, 72, 153, 0.6)',
    innerGradient: 'linear-gradient(145deg,#ec489933 0%,#f472b644 100%)',
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.23, 1, 0.32, 1],
    },
  },
};

export default function MemberPage() {
  const router = useRouter();

  const handleBack = () => {
    router.push('/');
  };

  return (
    <main className="min-h-screen relative overflow-hidden bg-[#0a0a0c] text-white">
      {/* Background - GridScan */}
      <div className="absolute inset-0 z-0 opacity-70 pointer-events-none">
        <GridScan
          sensitivity={0.55}
          lineThickness={1.5}
          linesColor="#4a3d6a"
          gridScale={0.12}
          scanColor="#4ade80"
          scanOpacity={0.5}
          bloomIntensity={0.8}
          noiseIntensity={0.02}
          scanDuration={2.5}
          scanDelay={2.0}
        />
      </div>
      <div className="absolute inset-0 z-[1] bg-gradient-to-b from-[#0a0a0c]/60 via-transparent to-[#0a0a0c]/80 pointer-events-none" />

      {/* Content */}
      <div className="relative z-10">
        <Header />

        <div className="max-w-7xl mx-auto px-4 lg:px-6 py-4 lg:py-6">
          {/* Back Button & Title */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-4 lg:mb-6"
          >
            <button
              onClick={handleBack}
              className="group flex items-center gap-2 text-white/60 hover:text-white transition-colors mb-3"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
              <span className="text-sm font-medium uppercase tracking-wide">Back</span>
            </button>

            <div className="flex items-center gap-3 mb-2">
              <Users className="w-6 h-6 text-emerald-400" />
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Team</h1>
            </div>
            <p className="text-white/50 text-sm max-w-2xl">
              AI Native Engineer 를 소개합니다.
            </p>
          </motion.div>

          {/* Team Grid - responsive auto-fit */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4 lg:gap-6 justify-items-center"
          >
            {TEAM_MEMBERS.map((member) => (
              <motion.div
                key={member.id}
                variants={itemVariants}
                className="w-full min-w-0 flex justify-center"
              >
                <ProfileCard
                  avatarUrl={member.avatarUrl}
                  miniAvatarUrl={member.miniAvatarUrl}
                  name={member.nameKr}
                  title={member.title}
                  handle={member.handle}
                  status={member.status}
                  showUserInfo={true}
                  githubUrl={member.githubUrl}
                  linkedinUrl={member.linkedinUrl}
                  behindGlowEnabled={true}
                  behindGlowColor={member.glowColor}
                  innerGradient={member.innerGradient}
                  enableTilt={true}
                  className="member-card w-full max-w-[260px]"
                />
              </motion.div>
            ))}
          </motion.div>

          {/* Footer Info */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 1 }}
            className="mt-6 pt-4 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-3"
          >
            <div className="flex items-center gap-6 text-white/40 text-sm">
              <a
                href="https://pseudo-lab.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white/70 transition-colors"
              >
                PseudoLab
              </a>
              <span className="text-white/20">•</span>
              <span>Agent Studio 11기</span>
              <span className="text-white/20">•</span>
              <span>Apache License 2.0</span>
            </div>
            <a
              href="https://github.com/Pseudo-Lab/Agent_Studio"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-white/40 hover:text-white/70 transition-colors"
            >
              <Github className="w-4 h-4" />
              <span className="text-sm">View on GitHub</span>
            </a>
          </motion.div>
        </div>
      </div>
    </main>
  );
}
