'use client';

import { ReactNode } from 'react';

// Simple provider wrapper (CopilotKit removed - using direct SSE API)
export function CopilotKitProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
