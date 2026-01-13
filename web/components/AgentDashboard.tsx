'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';
import ActionLog from './ActionLog';
import ThoughtPanel from './ThoughtPanel';
import StatePanel from './StatePanel';

interface AgentState {
  stage?: string;
  iteration?: number;
  thought?: string;
  action?: string;
  box_2d?: number[];
  adb_commands?: string[];
  status?: string;
  progress?: boolean;
  route?: string;
  max_iterations?: number;
  thought_history?: string[];
  analysis?: string;
}

interface AgentDashboardProps {
  agentState: AgentState;
  events: AgentState[];
  isRunning: boolean;
}

export default function AgentDashboard({ agentState, events, isRunning }: AgentDashboardProps) {
  // Extract data for each panel - use useMemo to avoid recalculation
  const actionLogs = useMemo(() => {
    // Include all events that have an action (any action)
    return events
      .filter((e) => e.action)
      .map((e, idx) => ({
        id: idx,
        iteration: e.iteration || 0,
        action: e.action || 'UNKNOWN',
        commands: e.adb_commands || [],
        status: e.status || 'unknown',
        timestamp: Date.now() + idx, // For unique keys
      }));
  }, [events]);

  const thoughts = useMemo(() => {
    // Include all events that have a thought
    return events
      .filter((e) => e.thought)
      .map((e, idx) => ({
        id: idx,
        iteration: e.iteration || 0,
        thought: e.thought || '',
        stage: e.stage || e.route || 'loop',
        timestamp: Date.now() + idx,
      }));
  }, [events]);

  // Current state with live updates from agentState
  const currentState = useMemo(() => {
    const lastEvent = events[events.length - 1];
    return {
      iteration: agentState.iteration ?? lastEvent?.iteration ?? 0,
      maxIterations: agentState.max_iterations || 20,
      status: agentState.status || lastEvent?.status || 'ready',
      progress: agentState.progress ?? lastEvent?.progress,
      route: agentState.route || agentState.stage || lastEvent?.route || lastEvent?.stage || 'idle',
      stage: agentState.stage || lastEvent?.stage || 'idle',
    };
  }, [agentState, events]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="grid grid-cols-1 lg:grid-cols-3 gap-5"
    >
      {/* Action Log Panel */}
      <div className="lg:col-span-1">
        <ActionLog logs={actionLogs} isRunning={isRunning} />
      </div>

      {/* Thought Panel */}
      <div className="lg:col-span-1">
        <ThoughtPanel thoughts={thoughts} isRunning={isRunning} />
      </div>

      {/* State Panel */}
      <div className="lg:col-span-1">
        <StatePanel state={currentState} isRunning={isRunning} />
      </div>
    </motion.div>
  );
}

