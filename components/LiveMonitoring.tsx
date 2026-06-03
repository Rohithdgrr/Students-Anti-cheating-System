
import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, Shield, Play, Square, Activity, Pause, Download, Video, Maximize2, Wifi, Eye, Brain } from 'lucide-react';
import { aiService, AIDetection } from '../frontend/services/aiService';
import { AlertLevel, ProctorAlert, DetectionStats } from '../types';
import { ClayCard } from './ClayCard';
import { ClayButton } from './ClayButton';

interface LiveMonitoringProps {
  onNewAlert: (alert: ProctorAlert) => void;
  updateIntegrityScore: (score: number) => void;
  updateStats: (stats: Omit<DetectionStats, 'expectedCount'>) => void;
  streamUrl?: string;
  isExternalStream?: boolean;
  cameraDeviceId?: string;
}

export const LiveMonitoring: React.FC<LiveMonitoringProps> = ({
  onNewAlert,
  updateIntegrityScore,
  updateStats,
  streamUrl,
  isExternalStream,
  cameraDeviceId
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiConnected, setAiConnected] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [useDirectCamera, setUseDirectCamera] = useState(true);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [mediapipeLoaded, setMediapipeLoaded] = useState(false);
  const [headPoseCount, setHeadPoseCount] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const [isRecording, setIsRecording] = useState(false);
  const pollTimerRef = useRef<number | null>(null);
  const browserStreamRef = useRef<MediaStream | null>(null);
  
  // Track captured violations to prevent duplicate screenshots (type + seat combo)
  const capturedViolationsRef = useRef<Map<string, number>>(new Map());
  const VIOLATION_COOLDOWN_MS = 10000; // 10 seconds cooldown per violation type per seat

  const checkAIConnection = useCallback(async () => {
    try {
      const details = await aiService.getHealthDetails();
      if (details) {
        setAiConnected(true);
        setMediapipeLoaded(details.mediapipe_loaded);
        return true;
      }
      setAiConnected(false);
      return false;
    } catch {
      setAiConnected(false);
      return false;
    }
  }, []);

  const handleScreenshot = useCallback(() => {
    if (!currentFrame) return;
    const link = document.createElement('a');
    link.download = `proctor-screenshot-${new Date().getTime()}.png`;
    link.href = `data:image/jpeg;base64,${currentFrame}`;
    link.click();
  }, [currentFrame]);

  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable full-screen mode: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  }, []);

  useEffect(() => {
    checkAIConnection();
    const interval = setInterval(checkAIConnection, 5000);
    return () => clearInterval(interval);
  }, [checkAIConnection]);

  const stopAll = useCallback(async () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    if (browserStreamRef.current) {
      browserStreamRef.current.getTracks().forEach(track => track.stop());
      browserStreamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }

    try {
      if (!useDirectCamera) {
        await aiService.stopStream();
      }
    } catch (e) {
      console.error('Error stopping:', e);
    }
    setCurrentFrame(null);
  }, [useDirectCamera]);

  const captureBrowserFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const isVideoReady =
      !!video &&
      !!canvas &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.videoWidth > 0 &&
      video.videoHeight > 0;

    if (!isVideoReady || !video || !canvas) {
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    if (!context) return null;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85).split(',')[1] ?? null;
  }, []);

  const pollFrames = useCallback(async (isDirect: boolean) => {
    try {
      if (isDirect) {
        const frame = captureBrowserFrame();
        if (!frame) return;

        const response = await aiService.analyzeFrame(frame);
        const annotatedFrame = response.annotated_frame || frame;
        setCurrentFrame(annotatedFrame);

        const integrityScore = aiService.calculateIntegrityScore(response);
        updateIntegrityScore(integrityScore);

        const stats = aiService.getDetectionStats(response);
        updateStats(stats);

        if (response.head_poses) {
          setHeadPoseCount(response.head_poses.length);
        }

        const alerts = aiService.convertDetectionsToAlerts(response);
        const now = Date.now();

        alerts.forEach(alertData => {
          const violationKey = `${alertData.type}-${alertData.seat}`;
          const lastCaptured = capturedViolationsRef.current.get(violationKey);

          let alertScreenshot = undefined;
          if (!lastCaptured || (now - lastCaptured) > VIOLATION_COOLDOWN_MS) {
            alertScreenshot = annotatedFrame;
            capturedViolationsRef.current.set(violationKey, now);
          }

          onNewAlert({
            ...alertData,
            id: crypto.randomUUID(),
            timestamp: new Date(),
            screenshot: alertScreenshot
          } as ProctorAlert);
        });
        return;
      }

      const response = await aiService.getFrame();
      if (!response) return;

      const { frame, result } = response;

      if (frame) {
        setCurrentFrame(frame);
      }

      if (result && frame) {
        const integrityScore = aiService.calculateIntegrityScore(result);
        updateIntegrityScore(integrityScore);

        const stats = aiService.getDetectionStats(result);
        updateStats(stats);

        if (result.head_poses) {
          setHeadPoseCount(result.head_poses.length);
        }

        const alerts = aiService.convertDetectionsToAlerts(result);
        const now = Date.now();

        alerts.forEach(alertData => {
          const violationKey = `${alertData.type}-${alertData.seat}`;
          const lastCaptured = capturedViolationsRef.current.get(violationKey);

          let alertScreenshot = undefined;
          if (!lastCaptured || (now - lastCaptured) > VIOLATION_COOLDOWN_MS) {
            alertScreenshot = frame;
            capturedViolationsRef.current.set(violationKey, now);
          }

          onNewAlert({
            ...alertData,
            id: crypto.randomUUID(),
            timestamp: new Date(),
            screenshot: alertScreenshot
          } as ProctorAlert);
        });
      }
    } catch (err) {
      // Frame not yet available, this is normal during startup
    }
  }, [captureBrowserFrame, onNewAlert, updateIntegrityScore, updateStats]);

  const startMonitoring = async () => {
    try {
      setDetectionError(null);
      setStreamError(null);

      const aiHealthy = await checkAIConnection();
      if (!aiHealthy) {
        setStreamError(`AI backend not responding at ${aiService.getBaseUrl()}. Please check your connection and verify the service is running.`);
        return;
      }

      if (!streamUrl || streamUrl === 'direct') {
        // LAPTOP CAMERA MODE: Capture from the browser webcam and send frames to the backend
        setUseDirectCamera(true);

        try {
          if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('Browser webcam is not supported in this environment');
          }

          let mediaStream: MediaStream;
          try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
              video: cameraDeviceId ? { deviceId: { exact: cameraDeviceId } } : true,
              audio: false,
            });
          } catch (error: unknown) {
            console.warn('Selected camera unavailable, falling back to the default browser camera.', error);
            setStreamError('Unable to access the selected camera. Using the default browser camera instead.');
            mediaStream = await navigator.mediaDevices.getUserMedia({
              video: true,
              audio: false,
            });
          }

          browserStreamRef.current = mediaStream;
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream;
            await videoRef.current.play();
          }
        } catch (e: any) {
          console.error('Failed to start browser webcam:', e);
          setStreamError(e?.message || 'Failed to start browser webcam');
          return;
        }

        setIsMonitoring(true);

        // Start polling for frames with detections
        pollTimerRef.current = window.setInterval(() => {
          pollFrames(true);
        }, 500);

        return;
      }

      // IP CAMERA / DROIDCAM MODE
      setUseDirectCamera(false);

      try {
        console.log('Starting IP camera stream:', streamUrl);
        await aiService.startStream(streamUrl);
        console.log('IP stream started on AI server');
      } catch (e: any) {
        console.error('Failed to start IP stream:', e);
        setStreamError(e?.message || 'Failed to start IP camera stream');
        return;
      }

      setIsMonitoring(true);

      // Start polling for frames
      pollTimerRef.current = window.setInterval(() => {
        pollFrames(false);
      }, 500);

    } catch (err: any) {
      console.error("Error starting monitoring:", err);
      setStreamError(err.message || "Failed to start camera");
    }
  };

  const stopMonitoring = async () => {
    await stopAll();
    setIsMonitoring(false);
    setIsPaused(false);
    setCurrentFrame(null);
    setHeadPoseCount(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
      if (browserStreamRef.current) {
        browserStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Activity className="text-[#6C5CE7]" /> {isExternalStream ? 'Remote Feed' : 'Direct Monitor'}
        </h2>
        <div className="flex gap-2 items-center">
          {!aiConnected && isMonitoring && (
            <span className="text-xs text-red-500 flex items-center gap-1">
              <Wifi size={12} /> AI Disconnected
            </span>
          )}
          {!isMonitoring ? (
            <ClayButton onClick={startMonitoring} variant="primary" className="px-4 py-2">
              <Play size={18} /> Start Session
            </ClayButton>
          ) : (
            <ClayButton onClick={stopMonitoring} variant="danger" className="px-4 py-2">
              <Square size={18} /> Stop
            </ClayButton>
          )}
        </div>
      </div>

      {streamError && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-2 rounded-xl text-sm mb-2">
          {streamError}
        </div>
      )}

      {detectionError && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm mb-2">
          {detectionError}
        </div>
      )}

      <div className="relative group" ref={containerRef}>
        <ClayCard className="relative overflow-hidden aspect-video bg-[#2D3436] flex items-center justify-center p-0 border-4 border-white shadow-xl">
          {!isMonitoring ? (
            <div className="text-center p-8">
              <div className="w-24 h-24 bg-[#F5F0EB]/10 rounded-full flex items-center justify-center mx-auto mb-6 clay-inset">
                <Camera size={48} className="text-[#636E72]" />
              </div>
              <p className="text-white/60 font-medium text-lg">Waiting for connection...</p>
              {!aiConnected && (
                <p className="text-red-400 text-sm mt-2">AI Server not responding</p>
              )}
              {aiConnected && (
                <div className="flex items-center justify-center gap-4 mt-3">
                  <span className="text-xs text-green-400 flex items-center gap-1">
                    <Eye size={12} /> YOLO Ready
                  </span>
                  {mediapipeLoaded && (
                    <span className="text-xs text-blue-400 flex items-center gap-1">
                      <Brain size={12} /> MediaPipe Ready
                    </span>
                  )}
                </div>
              )}
              <div className="mt-4 w-48 h-1 bg-white/10 rounded-full mx-auto overflow-hidden">
                <div className="w-1/3 h-full bg-[#6C5CE7] animate-[shimmer_1.5s_infinite]" />
              </div>
            </div>
          ) : (
            <div className="relative w-full h-full">
              {currentFrame ? (
                <img
                  ref={imgRef}
                  src={`data:image/jpeg;base64,${currentFrame}`}
                  alt="Live Feed with YOLO + MediaPipe detections"
                  className={`w-full h-full object-contain bg-black transition-opacity duration-500 ${isPaused ? 'opacity-40 grayscale' : 'opacity-100'}`}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-white/40 gap-4">
                  <div className="w-12 h-12 border-4 border-white/20 border-t-[#6C5CE7] rounded-full animate-spin" />
                  <p className="text-sm">Initializing camera &amp; AI models...</p>
                </div>
              )}
            </div>
          )}

          {isMonitoring && (
            <div className="absolute top-4 right-4 flex items-center gap-2">
              <div className="px-3 py-1 bg-white/90 backdrop-blur rounded-full clay-button text-[10px] font-bold flex items-center gap-2 shadow-lg">
                <span className={`w-1.5 h-1.5 rounded-full ${aiConnected ? 'bg-[#00B894]' : 'bg-red-500'} ${currentFrame ? 'animate-pulse' : ''}`} />
                {aiConnected ? (currentFrame ? 'LIVE' : 'CONNECTING...') : 'AI OFFLINE'}
              </div>
              {headPoseCount > 0 && (
                <div className="px-3 py-1 bg-blue-50/90 backdrop-blur rounded-full text-[10px] font-bold flex items-center gap-2 shadow-lg text-blue-600">
                  <Brain size={10} />
                  {headPoseCount} FACE{headPoseCount > 1 ? 'S' : ''}
                </div>
              )}
            </div>
          )}

          {isPaused && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="bg-white/20 backdrop-blur-md p-6 rounded-full">
                <Pause size={48} className="text-white" />
              </div>
            </div>
          )}
        </ClayCard>

        {isMonitoring && (
          <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-white/90 backdrop-blur-md p-2 rounded-[24px] clay-card shadow-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <button
              onClick={() => {
                setIsPaused(!isPaused);
                if (!isPaused && pollTimerRef.current) {
                  clearInterval(pollTimerRef.current);
                  pollTimerRef.current = null;
                } else if (isPaused) {
                  pollTimerRef.current = window.setInterval(() => {
                    pollFrames(useDirectCamera);
                  }, 500);
                }
              }}
              className={`w-10 h-10 rounded-full flex items-center justify-center transition-colors ${isPaused ? 'bg-[#6C5CE7] text-white' : 'text-[#636E72] hover:bg-[#F5F0EB]'}`}
              title={isPaused ? "Resume" : "Pause"}
            >
              {isPaused ? <Play size={18} /> : <Pause size={18} />}
            </button>
            <div className="w-[1px] h-6 bg-[#E8E2DC]" />
            <button
              onClick={handleScreenshot}
              className="w-10 h-10 rounded-full flex items-center justify-center text-[#636E72] hover:bg-[#F5F0EB] transition-colors" title="Screenshot"
            >
              <Download size={18} />
            </button>
            <button
              onClick={toggleFullscreen}
              className="w-10 h-10 rounded-full flex items-center justify-center text-[#636E72] hover:bg-[#F5F0EB] transition-colors" title="Fullscreen"
            >
              <Maximize2 size={18} />
            </button>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />
      <video ref={videoRef} className="hidden" playsInline muted autoPlay />

      {isMonitoring && (
        <ClayCard className="bg-[#6C5CE7]/5 border-[#6C5CE7]/20 flex items-start gap-4">
          <div className="p-3 bg-[#6C5CE7] rounded-xl clay-button shrink-0">
            <Shield className="text-white" size={24} />
          </div>
          <div>
            <h3 className="font-bold text-[#6C5CE7] text-sm">AI Active Protection</h3>
            <p className="text-xs text-[#2D3436]/80 leading-relaxed">
              {aiConnected
                ? <>
                  <span className="font-semibold">YOLO</span> scanning for prohibited items (phones, headphones, books).{' '}
                  {mediapipeLoaded && (
                    <><span className="font-semibold">MediaPipe</span> tracking head pose (yaw/pitch/roll) for attention monitoring. </>
                  )}
                </>
                : `AI Server disconnected. Check ${aiService.getBaseUrl()}`}
              <span className="font-bold ml-2">Last check: {new Date().toLocaleTimeString()}</span>
            </p>
          </div>
        </ClayCard>
      )}
    </div>
  );
};
