
import React, { useState, useEffect } from 'react';
import { Wifi, Link as LinkIcon, Info, Camera, Smartphone, RefreshCw, Monitor } from 'lucide-react';
import { ClayCard } from '../ClayCard';
import { ClayButton } from '../ClayButton';
import { aiService, CameraDevice } from '../../frontend/services/aiService';

interface ConnectionPanelProps {
  onConnect: (url: string, source: 'direct' | 'ip', cameraDeviceId?: string) => void;
  status: 'connected' | 'disconnected' | 'connecting';
}

export const ConnectionPanel: React.FC<ConnectionPanelProps> = ({ onConnect, status }) => {
  const [url, setUrl] = useState('http://192.168.1.5:4747/video');
  const [source, setSource] = useState<'direct' | 'ip'>('direct');
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [loadingCameras, setLoadingCameras] = useState(false);

  const refreshCameras = async () => {
    setLoadingCameras(true);
    try {
      const cams = await aiService.listCameras();
      setCameras(cams);
      if (cams.length > 0) {
        setSelectedCamera(cams[0].deviceId);
      }
    } catch (e) {
      console.error('Failed to list cameras:', e);
    }
    setLoadingCameras(false);
  };

  useEffect(() => {
    refreshCameras();
  }, []);

  return (
    <ClayCard className="bg-white">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold flex items-center gap-2 text-[#2D3436]">
          <Wifi size={20} className="text-[#6C5CE7]" />
          📡 Camera Connection
        </h3>
        <div className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest flex items-center gap-2 ${status === 'connected' ? 'bg-[#00B894]/10 text-[#00B894]' :
            status === 'connecting' ? 'bg-[#FFA502]/10 text-[#FFA502]' :
              'bg-[#FF6B6B]/10 text-[#FF6B6B]'
          }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${status === 'connected' ? 'bg-[#00B894] shadow-[0_0_8px_#00B894]' :
              status === 'connecting' ? 'bg-[#FFA502] animate-pulse' :
                'bg-[#FF6B6B]'
            }`} />
          {status}
        </div>
      </div>

      <div className="space-y-4">
        {/* Camera Source Toggle */}
        <div className="flex bg-[#F5F0EB] p-1 rounded-xl">
          <button
            onClick={() => setSource('direct')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium transition-all ${source === 'direct'
                ? 'bg-white text-[#6C5CE7] shadow-sm'
                : 'text-[#636E72] hover:text-[#2D3436]'
              }`}
          >
            <Camera size={14} />
            Browser Webcam
          </button>
          <button
            onClick={() => setSource('ip')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium transition-all ${source === 'ip'
                ? 'bg-white text-[#6C5CE7] shadow-sm'
                : 'text-[#636E72] hover:text-[#2D3436]'
              }`}
          >
            <Smartphone size={14} />
            IP Camera
          </button>
        </div>

        {source === 'direct' && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-[10px] font-bold text-[#636E72] uppercase tracking-wider ml-1">Available Cameras</label>
              <button
                onClick={refreshCameras}
                disabled={loadingCameras}
                className="text-[#6C5CE7] hover:bg-[#6C5CE7]/10 p-1 rounded-lg transition-colors"
                title="Refresh cameras"
              >
                <RefreshCw size={12} className={loadingCameras ? 'animate-spin' : ''} />
              </button>
            </div>
            {cameras.length > 0 ? (
              <div className="space-y-1.5">
                {cameras.map(cam => (
                  <button
                    key={cam.index}
                    onClick={() => setSelectedCamera(cam.deviceId)}
                    className={`w-full flex items-center gap-3 p-2.5 rounded-xl text-left text-xs transition-all ${selectedCamera === cam.deviceId
                        ? 'bg-[#6C5CE7]/10 text-[#6C5CE7] border border-[#6C5CE7]/30'
                        : 'bg-[#F5F0EB]/50 text-[#636E72] hover:bg-[#F5F0EB] border border-transparent'
                      }`}
                  >
                    <Monitor size={14} />
                    <div>
                      <span className="font-semibold block">{cam.name}</span>
                      <span className="text-[10px] opacity-70">{cam.resolution}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-xs text-[#636E72] bg-[#F5F0EB] p-3 rounded-xl text-center">
                {loadingCameras ? 'Scanning cameras...' : 'No cameras detected. Click refresh.'}
              </div>
            )}
          </div>
        )}

        {source === 'ip' && (
          <div className="relative">
            <label className="text-[10px] font-bold text-[#636E72] uppercase tracking-wider mb-1 block ml-1">Stream URL</label>
            <div className="clay-inset p-1 flex items-center gap-2">
              <div className="pl-3 text-[#636E72]">
                <LinkIcon size={14} />
              </div>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="w-full bg-transparent border-none focus:ring-0 text-sm mono py-2 text-[#2D3436]"
                placeholder="http://ip-address:port/video"
              />
            </div>
            <p className="text-[10px] text-[#636E72] mt-1.5 ml-1">
              DroidCam: <span className="mono text-[#6C5CE7]">http://phone-ip:4747/video</span>
            </p>
          </div>
        )}

        <ClayButton
          variant={status === 'connected' ? 'ghost' : 'primary'}
          className="w-full"
          onClick={() => onConnect(source === 'ip' ? url : 'direct', source, source === 'direct' ? selectedCamera || undefined : undefined)}
          disabled={status === 'connecting'}
        >
          {status === 'connected' ? 'Reconnect Camera' : source === 'direct' ? 'Start Webcam' : 'Connect IP Camera'}
        </ClayButton>

        <div className="bg-[#F5F0EB] p-3 rounded-xl flex gap-3 items-start">
          <Info size={16} className="text-[#6C5CE7] shrink-0 mt-0.5" />
          <p className="text-[11px] text-[#636E72] leading-relaxed">
            {source === 'direct' ? (
              <><span className="font-bold text-[#2D3436]">Direct Mode:</span> Uses your browser webcam, sends frames to the Render backend, and returns live YOLO + MediaPipe detections.</>
            ) : (
              <><span className="font-bold text-[#2D3436]">IP Camera Mode:</span> Open DroidCam/IP Webcam app on your phone → Start server → Enter the URL shown in the app. Supports MJPEG streams.</>
            )}
          </p>
        </div>
      </div>
    </ClayCard>
  );
};
