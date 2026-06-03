
import React, { useState } from 'react';
import { ShieldCheck, Timer, Users, Activity } from 'lucide-react';
import { LiveMonitoring } from '../components/LiveMonitoring';
import { AlertHistory } from '../components/AlertHistory';
import { ClayCard } from '../components/ClayCard';
import { ConnectionPanel } from '../components/live/ConnectionPanel';
import { DetectionStats } from '../components/live/DetectionStats';
import { ScoreTable } from '../components/live/ScoreTable';
import { EvidenceSection } from '../components/live/EvidenceSection';
import { NotificationPanel } from '../components/NotificationPanel';
import { ProctorAlert, DetectionStats as IStats } from '../types';
import { COLORS } from '../constants';

interface LiveMonitorViewProps {
  alerts: ProctorAlert[];
  integrityScore: number;
  handleNewAlert: (alert: ProctorAlert) => void;
  updateIntegrityScore: (score: number) => void;
  updateStats?: (stats: Omit<IStats, 'expectedCount'>) => void;
}

export const LiveMonitorView: React.FC<LiveMonitorViewProps> = ({
  alerts,
  integrityScore,
  handleNewAlert,
  updateIntegrityScore,
  updateStats: updateStatsProp,
}) => {
  const [connStatus, setConnStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [streamUrl, setStreamUrl] = useState('');
  const [cameraSource, setCameraSource] = useState<'direct' | 'ip'>('direct');
  const [cameraDeviceId, setCameraDeviceId] = useState('');
  const [stats, setStats] = useState<IStats>({
    phone: 0,
    chit: 0,
    textbook: 0,
    notebook: 0,
    device: 0,
    headTurn: 0,
    leaning: 0,
    multiplePeople: 0,
    detectedCount: 0,
    expectedCount: 0
  });
  const [notificationAlert, setNotificationAlert] = useState<ProctorAlert | null>(null);

  const handleConnect = (url: string, source: 'direct' | 'ip', deviceId?: string) => {
    setStreamUrl(url);
    setCameraSource(source);
    setCameraDeviceId(deviceId || '');
    setConnStatus('connecting');
    // Simulate connection delay
    setTimeout(() => setConnStatus('connected'), 1500);
  };

  const updateStats = (newStats: Omit<IStats, 'expectedCount'>) => {
    setStats(prev => ({ ...prev, ...newStats }));
    // Propagate to parent (App.tsx) for invigilator view
    updateStatsProp?.(newStats);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 pb-10">

      {/* LEFT COLUMN: Controls & Stats */}
      <div className="lg:col-span-3 space-y-6">
        <ConnectionPanel status={connStatus} onConnect={handleConnect} />
        <DetectionStats stats={stats} />
      </div>

      {/* CENTER COLUMN: Live Feed & Performance */}
      <div className="lg:col-span-6 space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <ClayCard className="flex items-center gap-4 bg-white p-4">
            <div className="p-3 bg-[#6C5CE7]/10 rounded-2xl shrink-0">
              <ShieldCheck className="text-[#6C5CE7]" size={24} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-[#636E72] uppercase tracking-wider">Integrity</p>
              <p className="text-xl font-bold mono" style={{ color: integrityScore > 80 ? COLORS.success : COLORS.danger }}>
                {integrityScore}%
              </p>
            </div>
          </ClayCard>
          <ClayCard className="flex items-center gap-4 bg-white p-4">
            <div className="p-3 bg-[#FFA502]/10 rounded-2xl shrink-0">
              <Timer className="text-[#FFA502]" size={24} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-[#636E72] uppercase tracking-wider">Elapsed</p>
              <p className="text-xl font-bold mono text-[#2D3436]">00:00:00</p>
            </div>
          </ClayCard>
        </div>

        <ClayCard className="p-8 bg-white" delay={0.1}>
          <LiveMonitoring
            onNewAlert={handleNewAlert}
            updateIntegrityScore={updateIntegrityScore}
            updateStats={updateStats}
            streamUrl={streamUrl}
            isExternalStream={cameraSource === 'ip'}
            cameraDeviceId={cameraDeviceId}
          />
        </ClayCard>

        <ScoreTable alerts={alerts} />
        <EvidenceSection alerts={alerts} />
      </div>

      {/* RIGHT COLUMN: Alert Feed */}
      <div className="lg:col-span-3 h-[calc(100vh-160px)] sticky top-28">
        <ClayCard className="h-full bg-white flex flex-col p-5 overflow-hidden">
          <AlertHistory alerts={alerts} onNotifyInvigilator={setNotificationAlert} />
        </ClayCard>
      </div>

      {/* Notification Panel */}
      <NotificationPanel
        isOpen={!!notificationAlert}
        onClose={() => setNotificationAlert(null)}
        violationType={notificationAlert?.type}
        seat={notificationAlert?.seat}
        confidence={notificationAlert?.confidence}
      />
    </div>
  );
};
