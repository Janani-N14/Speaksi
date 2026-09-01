/**
 * Speaky AI - Pronunciation Checker Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const targetTextInput = document.getElementById('target-text');
    const btnClearText = document.getElementById('btn-clear-text');
    const btnListenTarget = document.getElementById('btn-listen-target');
    const targetTtsPlayer = document.getElementById('target-tts-player');
    const presetChips = document.querySelectorAll('.preset-chips .chip');

    // Tabs
    const tabMic = document.getElementById('tab-mic');
    const tabFile = document.getElementById('tab-file');
    const panelMic = document.getElementById('panel-mic');
    const panelFile = document.getElementById('panel-file');

    // Recorder Elements
    const btnStartRecord = document.getElementById('btn-start-record');
    const btnStopRecord = document.getElementById('btn-stop-record');
    const btnResetRecord = document.getElementById('btn-reset-record');
    const recordingTimer = document.getElementById('recording-time');
    const micStatusLabel = document.getElementById('mic-status-label');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const recordedPreviewBox = document.getElementById('recorded-preview-box');
    const audioPlayer = document.getElementById('audio-player');

    // File Upload Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectedFileInfo = document.getElementById('selected-file-info');
    const selectedFilename = document.getElementById('selected-filename');

    // Submit & Results Elements
    const btnAnalyze = document.getElementById('btn-analyze');
    const resultsEmptyState = document.getElementById('results-empty-state');
    const resultsError = document.getElementById('results-error');
    const errorMessageText = document.getElementById('error-message-text');
    const resultsContent = document.getElementById('results-content');
    const scoreStatusPill = document.getElementById('score-status-pill');

    // Results Sub-Elements
    const gaugeProgress = document.getElementById('gauge-progress');
    const gaugeNumber = document.getElementById('gauge-number');
    const targetDisplay = document.getElementById('target-display');
    const transcriptionDisplay = document.getElementById('transcription-display');
    const mismatchTags = document.getElementById('mismatch-tags');
    const therapistTipsBody = document.getElementById('therapist-tips-body');
    const videoGuideSection = document.getElementById('video-guide-section');
    const videoButtonsContainer = document.getElementById('video-buttons-container');
    const videoFrameWrap = document.getElementById('video-frame-wrap');
    const videoIframe = document.getElementById('video-iframe');
    const rawJsonViewer = document.getElementById('raw-json-viewer');

    // State Variables
    let currentMode = 'mic'; // 'mic' | 'file'
    let mediaRecorder = null;
    let audioStream = null;
    let audioContext = null;
    let analyserNode = null;
    let animationFrameId = null;
    let recordedChunks = [];
    let recordedBlob = null;
    let selectedUploadedFile = null;
    let recordStartTime = 0;
    let timerInterval = null;

    // Canvas setup for visualizer
    const canvasCtx = waveformCanvas.getContext('2d');

    /* ==========================================================================
       1. Target Text & Preset Chips
       ========================================================================== */
    presetChips.forEach(chip => {
        chip.addEventListener('click', () => {
            presetChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            targetTextInput.value = chip.dataset.word;
        });
    });

    btnClearText.addEventListener('click', () => {
        targetTextInput.value = '';
        targetTextInput.focus();
        presetChips.forEach(c => c.classList.remove('active'));
    });

    // Listen to Target Audio (TTS Preview)
    btnListenTarget.addEventListener('click', async () => {
        const text = targetTextInput.value.trim();
        if (!text) {
            alert('Please enter a word or phrase to listen to.');
            return;
        }

        btnListenTarget.disabled = true;
        btnListenTarget.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';

        try {
            targetTtsPlayer.src = `/tts?text=${encodeURIComponent(text)}&t=${Date.now()}`;
            await targetTtsPlayer.play();
        } catch (err) {
            console.error('TTS playback error:', err);
        } finally {
            btnListenTarget.disabled = false;
            btnListenTarget.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen Target';
        }
    });

    /* ==========================================================================
       2. Input Mode Switching
       ========================================================================== */
    tabMic.addEventListener('click', () => {
        currentMode = 'mic';
        tabMic.classList.add('active');
        tabFile.classList.remove('active');
        panelMic.style.display = 'flex';
        panelFile.style.display = 'none';
    });

    tabFile.addEventListener('click', () => {
        currentMode = 'file';
        tabFile.classList.add('active');
        tabMic.classList.remove('active');
        panelFile.style.display = 'block';
        panelMic.style.display = 'none';
    });

    /* ==========================================================================
       3. Live Microphone Recording & Waveform Visualizer
       ========================================================================== */
    function resizeCanvas() {
        waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
        waveformCanvas.height = waveformCanvas.parentElement.clientHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function drawWaveform() {
        if (!analyserNode) return;

        animationFrameId = requestAnimationFrame(drawWaveform);
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserNode.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = '#060913';
        canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

        canvasCtx.lineWidth = 2.5;
        const gradient = canvasCtx.createLinearGradient(0, 0, waveformCanvas.width, 0);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(0.5, '#06b6d4');
        gradient.addColorStop(1, '#10b981');
        canvasCtx.strokeStyle = gradient;

        canvasCtx.beginPath();
        const sliceWidth = waveformCanvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * (waveformCanvas.height / 2);

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
        canvasCtx.stroke();
    }

    function drawIdleLine() {
        canvasCtx.fillStyle = '#060913';
        canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        canvasCtx.lineWidth = 1.5;
        canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, waveformCanvas.height / 2);
        canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
        canvasCtx.stroke();
    }
    drawIdleLine();

    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    btnStartRecord.addEventListener('click', async () => {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Audio Context for Live Waveform
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(audioStream);
            analyserNode = audioContext.createAnalyser();
            analyserNode.fftSize = 2048;
            source.connect(analyserNode);

            recordedChunks = [];
            let options = {};
            if (MediaRecorder.isTypeSupported('audio/webm')) {
                options.mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/wav')) {
                options.mimeType = 'audio/wav';
            }

            mediaRecorder = new MediaRecorder(audioStream, options);
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const mimeType = mediaRecorder.mimeType || 'audio/webm';
                recordedBlob = new Blob(recordedChunks, { type: mimeType });
                const audioUrl = URL.createObjectURL(recordedBlob);
                audioPlayer.src = audioUrl;
                recordedPreviewBox.style.display = 'flex';
                btnResetRecord.style.display = 'inline-flex';
                micStatusLabel.textContent = 'Audio recorded successfully! Ready to analyze.';

                // Stop tracks
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                if (audioContext) {
                    audioContext.close();
                }
                cancelAnimationFrame(animationFrameId);
                drawIdleLine();
            };

            mediaRecorder.start(100);
            recordStartTime = Date.now();
            recordingTimer.textContent = '00:00';
            timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
                recordingTimer.textContent = formatTime(elapsed);
            }, 1000);

            // UI State
            btnStartRecord.disabled = true;
            btnStartRecord.classList.add('recording');
            btnStartRecord.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Recording...';
            btnStopRecord.disabled = false;
            btnResetRecord.style.display = 'none';
            micStatusLabel.textContent = 'Listening... Speak your target phrase now';

            drawWaveform();

        } catch (err) {
            console.error('Microphone error:', err);
            alert('Could not access microphone. Please check your browser microphone permissions.');
        }
    });

    btnStopRecord.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            clearInterval(timerInterval);

            btnStartRecord.disabled = false;
            btnStartRecord.classList.remove('recording');
            btnStartRecord.innerHTML = '<i class="fa-solid fa-circle"></i> Start Recording';
            btnStopRecord.disabled = true;
        }
    });

    btnResetRecord.addEventListener('click', () => {
        recordedBlob = null;
        recordedChunks = [];
        audioPlayer.src = '';
        recordedPreviewBox.style.display = 'none';
        btnResetRecord.style.display = 'none';
        recordingTimer.textContent = '00:00';
        micStatusLabel.textContent = 'Click "Start Recording" and speak clearly';
        drawIdleLine();
    });

    /* ==========================================================================
       4. File Upload & Drag-and-Drop
       ========================================================================== */
    dropZone.addEventListener('click', (e) => {
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    function handleSelectedFile(file) {
        selectedUploadedFile = file;
        selectedFilename.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileInfo.style.display = 'inline-flex';
    }

    /* ==========================================================================
       5. Analysis Submission & Results Rendering
       ========================================================================== */
    btnAnalyze.addEventListener('click', async () => {
        const text = targetTextInput.value.trim();
        if (!text) {
            alert('Please enter a target phrase or word to evaluate.');
            targetTextInput.focus();
            return;
        }

        let audioToUpload = null;
        let filename = 'recording.webm';

        if (currentMode === 'mic') {
            if (!recordedBlob) {
                alert('Please record your pronunciation first before analyzing.');
                return;
            }
            audioToUpload = recordedBlob;
            filename = 'mic_recording.webm';
        } else {
            if (!selectedUploadedFile) {
                alert('Please select or drop an audio file first.');
                return;
            }
            audioToUpload = selectedUploadedFile;
            filename = selectedUploadedFile.name;
        }

        // Prepare FormData
        const formData = new FormData();
        formData.append('text', text);
        formData.append('audio', audioToUpload, filename);

        // Show Loading State
        btnAnalyze.disabled = true;
        btnAnalyze.querySelector('.btn-text').style.display = 'none';
        btnAnalyze.querySelector('.btn-loader').style.display = 'inline-flex';
        resultsError.style.display = 'none';

        try {
            const response = await fetch('/check-pronunciation', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server responded with HTTP ${response.status}`);
            }

            renderResults(data);

        } catch (err) {
            console.error('Evaluation Error:', err);
            resultsEmptyState.style.display = 'none';
            resultsContent.style.display = 'none';
            resultsError.style.display = 'flex';
            errorMessageText.textContent = err.message || 'An unexpected error occurred during pronunciation analysis.';
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.querySelector('.btn-text').style.display = 'inline-flex';
            btnAnalyze.querySelector('.btn-loader').style.display = 'none';
        }
    });

    function renderResults(data) {
        resultsEmptyState.style.display = 'none';
        resultsError.style.display = 'none';
        resultsContent.style.display = 'flex';

        // 1. Gauge Progress Score
        const score = Math.round(data.similarity || 0);
        gaugeNumber.textContent = score;

        // Circular Gauge circumference = 2 * PI * 50 = 314.16
        const circumference = 314.16;
        const offset = circumference - (score / 100) * circumference;
        gaugeProgress.style.strokeDashoffset = offset;

        // Color coding by score
        let scoreClass = 'tag-perfect';
        let statusText = 'Excellent Match';
        let strokeColor = '#10b981';

        if (score >= 80) {
            scoreClass = 'tag-perfect';
            statusText = '🌟 Excellent';
            strokeColor = '#10b981';
        } else if (score >= 60) {
            scoreClass = 'tag-mismatch';
            statusText = '👍 Good Effort';
            strokeColor = '#f59e0b';
        } else {
            scoreClass = 'tag-mismatch';
            statusText = '🎯 Needs Practice';
            strokeColor = '#f43f5e';
        }

        gaugeProgress.style.stroke = strokeColor;
        scoreStatusPill.textContent = `${score}% • ${statusText}`;
        scoreStatusPill.className = `badge ${score >= 80 ? 'badge-status' : 'badge-endpoint'}`;

        // 2. Comparison Box (Target vs Transcription)
        targetDisplay.textContent = data.target_text || targetTextInput.value;
        transcriptionDisplay.textContent = data.transcription ? `"${data.transcription}"` : '(No speech detected)';

        // 3. Mismatched Sounds Tags
        mismatchTags.innerHTML = '';
        const mismatches = data.mis_matchings || [];

        if (mismatches.length === 0 && (data.transcription || '').toLowerCase() === (data.target_text || '').toLowerCase()) {
            mismatchTags.innerHTML = '<span class="tag tag-perfect"><i class="fa-solid fa-circle-check"></i> Perfect Pronunciation!</span>';
        } else if (mismatches.length === 0) {
            mismatchTags.innerHTML = '<span class="tag tag-perfect"><i class="fa-solid fa-check"></i> All sounds recognized</span>';
        } else {
            mismatches.forEach(letter => {
                const tag = document.createElement('span');
                tag.className = 'tag tag-mismatch';
                tag.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> Sound '${letter}'`;
                mismatchTags.appendChild(tag);
            });
        }

        // 4. AI Therapist Tips (Speaky)
        if (window.marked) {
            therapistTipsBody.innerHTML = marked.parse(data.tips || 'No advice generated.');
        } else {
            therapistTipsBody.textContent = data.tips || 'No advice generated.';
        }

        // 5. Video Guides
        const videos = data.videos || [];
        if (videos.length > 0) {
            videoGuideSection.style.display = 'block';
            videoButtonsContainer.innerHTML = '';
            videoFrameWrap.style.display = 'none';

            videos.forEach((vid, idx) => {
                const chipBtn = document.createElement('button');
                chipBtn.type = 'button';
                chipBtn.className = `video-chip-btn ${idx === 0 ? 'active' : ''}`;
                chipBtn.innerHTML = `<i class="fa-brands fa-youtube"></i> Sound '${vid.phoneme}' (${vid.start_time}s)`;
                
                chipBtn.addEventListener('click', () => {
                    document.querySelectorAll('.video-chip-btn').forEach(b => b.classList.remove('active'));
                    chipBtn.classList.add('active');
                    videoIframe.src = vid.url;
                    videoFrameWrap.style.display = 'block';
                });

                videoButtonsContainer.appendChild(chipBtn);
            });

            // Automatically load first video
            if (videos[0]) {
                videoIframe.src = videos[0].url;
                videoFrameWrap.style.display = 'block';
            }
        } else {
            videoGuideSection.style.display = 'none';
        }

        // 6. Raw JSON Viewer
        rawJsonViewer.textContent = JSON.stringify(data, null, 2);
    }
});
