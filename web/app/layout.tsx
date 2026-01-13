import type { Metadata } from 'next';
import './globals.css';
import '../components/ProfileCard.css';
import { CopilotKitProvider } from './providers';

export const metadata: Metadata = {
  title: 'Agent Studio | 가짜연구소',
  description: 'Pseudo Lab | Agent Studio — 빅맥 주문 데모',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko" className="dark">
      <body className="min-h-screen bg-pseudo-darker grid-pattern">
        <CopilotKitProvider>
          {children}
        </CopilotKitProvider>
      </body>
    </html>
  );
}

