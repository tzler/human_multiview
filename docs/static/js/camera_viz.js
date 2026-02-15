/* ============================================================
   3D Camera Trajectory Visualization (Three.js)
   Always synced to headcam video time. Loops automatically.
   ============================================================ */

let cameraVizScene, cameraVizCamera, cameraVizRenderer, cameraVizControls;
let cameraFrustums = [];
let cameraVizData = null;

// Sync state
let depthDisplay = null;
let headcamVideo = null;
let lastSyncedIdx = -1;

async function initCameraViz() {
    const container = document.getElementById('camera-viz-3d');
    if (!container || typeof THREE === 'undefined') return;

    depthDisplay = document.getElementById('depth-display');
    headcamVideo = document.getElementById('headcam-replay');

    // Load camera data
    try {
        const resp = await fetch('static/data/headcam_cameras/camera_poses.json');
        cameraVizData = await resp.json();
    } catch (e) {
        console.warn('Could not load camera poses', e);
        return;
    }

    // Preload depth images
    for (let i = 0; i < cameraVizData.n_frames; i++) {
        const img = new Image();
        img.src = `static/data/headcam_cameras/depths/depth_${String(i).padStart(2, '0')}.jpg`;
    }

    const width = container.clientWidth;
    const height = container.clientHeight || 280;

    // Scene
    cameraVizScene = new THREE.Scene();
    cameraVizScene.background = new THREE.Color(0xf7f5f2);

    // Camera
    cameraVizCamera = new THREE.PerspectiveCamera(50, width / height, 0.01, 50);

    // Renderer
    cameraVizRenderer = new THREE.WebGLRenderer({ antialias: true });
    cameraVizRenderer.setSize(width, height);
    cameraVizRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(cameraVizRenderer.domElement);

    // Orbit controls — no auto-rotate
    cameraVizControls = new THREE.OrbitControls(cameraVizCamera, cameraVizRenderer.domElement);
    cameraVizControls.enableDamping = true;
    cameraVizControls.dampingFactor = 0.05;
    cameraVizControls.autoRotate = false;
    cameraVizControls.maxDistance = 8;
    cameraVizControls.minDistance = 0.5;

    // Subtle ground grid
    const gridHelper = new THREE.GridHelper(3, 12, 0xdddddd, 0xeeeeee);
    gridHelper.position.y = -0.5;
    cameraVizScene.add(gridHelper);

    // Ambient light
    cameraVizScene.add(new THREE.AmbientLight(0xffffff, 0.8));

    // Trajectory path
    buildTrajectoryPath();

    // Camera frustums
    await buildCameraFrustums();

    // Center scene
    centerScene();

    // Start render loop
    animateCameraViz();

    // Resize handler
    const resizeObs = new ResizeObserver(() => {
        const w = container.clientWidth;
        const h = container.clientHeight || 280;
        cameraVizCamera.aspect = w / h;
        cameraVizCamera.updateProjectionMatrix();
        cameraVizRenderer.setSize(w, h);
    });
    resizeObs.observe(container);
}

function buildTrajectoryPath() {
    if (!cameraVizData) return;
    const positions = cameraVizData.camera_positions;
    const points = positions.map(p => new THREE.Vector3(p[0], -p[1], -p[2]));

    const curve = new THREE.CatmullRomCurve3(points);
    const curvePoints = curve.getPoints(100);

    const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
    const material = new THREE.LineBasicMaterial({
        color: 0xcccccc, transparent: true, opacity: 0.6,
    });
    cameraVizScene.add(new THREE.Line(geometry, material));
}

async function buildCameraFrustums() {
    const n = cameraVizData.n_frames;
    const positions = cameraVizData.camera_positions;
    const extrinsics = cameraVizData.extrinsics;

    const colorStart = new THREE.Color(0x2a9d8f);
    const colorEnd = new THREE.Color(0xe9c46a);
    const textureLoader = new THREE.TextureLoader();

    for (let i = 0; i < n; i++) {
        const group = new THREE.Group();
        const t = n > 1 ? i / (n - 1) : 0;
        const color = new THREE.Color().lerpColors(colorStart, colorEnd, t);

        const pos = new THREE.Vector3(positions[i][0], -positions[i][1], -positions[i][2]);

        const R = extrinsics[i];
        const rotMatrix = new THREE.Matrix4();
        rotMatrix.set(
            R[0][0], -R[0][1], -R[0][2], 0,
            -R[1][0], R[1][1], R[1][2], 0,
            -R[2][0], R[2][1], R[2][2], 0,
            0, 0, 0, 1
        );
        const invRot = rotMatrix.clone().invert();

        const frustumSize = 0.06;
        const aspect = 480 / 854;
        const hw = frustumSize * aspect;
        const hh = frustumSize;
        const depth = frustumSize * 1.2;

        const corners = [
            new THREE.Vector3(-hw, -hh, -depth),
            new THREE.Vector3(hw, -hh, -depth),
            new THREE.Vector3(hw, hh, -depth),
            new THREE.Vector3(-hw, hh, -depth),
        ];
        corners.forEach(c => c.applyMatrix4(invRot));

        const lineVerts = [];
        const origin = new THREE.Vector3(0, 0, 0);
        for (const c of corners) {
            lineVerts.push(origin.x, origin.y, origin.z, c.x, c.y, c.z);
        }
        for (let j = 0; j < 4; j++) {
            const c1 = corners[j], c2 = corners[(j + 1) % 4];
            lineVerts.push(c1.x, c1.y, c1.z, c2.x, c2.y, c2.z);
        }

        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lineVerts, 3));
        const wireframe = new THREE.LineSegments(lineGeo,
            new THREE.LineBasicMaterial({ color: color }));

        const planeGeo = new THREE.PlaneGeometry(hw * 2, hh * 2);
        const planeCenter = new THREE.Vector3(0, 0, -depth);
        planeCenter.applyMatrix4(invRot);

        const planeMesh = new THREE.Mesh(planeGeo, new THREE.MeshBasicMaterial({
            color: 0xffffff, transparent: true, opacity: 0.9, side: THREE.DoubleSide,
        }));
        planeMesh.position.copy(planeCenter);
        const quat = new THREE.Quaternion();
        quat.setFromRotationMatrix(invRot);
        planeMesh.quaternion.copy(quat);

        const frameIdx = String(i).padStart(2, '0');
        textureLoader.load(
            `static/data/headcam_cameras/frames/frame_${frameIdx}.jpg`,
            (texture) => {
                texture.colorSpace = THREE.SRGBColorSpace;
                planeMesh.material.map = texture;
                planeMesh.material.needsUpdate = true;
            }
        );

        group.add(wireframe);
        group.add(planeMesh);
        group.position.copy(pos);
        group.visible = false;

        cameraVizScene.add(group);
        cameraFrustums.push(group);
    }
}

function centerScene() {
    if (!cameraVizData) return;
    const positions = cameraVizData.camera_positions;
    const center = new THREE.Vector3();
    positions.forEach(p => center.add(new THREE.Vector3(p[0], -p[1], -p[2])));
    center.divideScalar(positions.length);

    cameraVizControls.target.copy(center);

    const bbox = new THREE.Box3();
    positions.forEach(p => bbox.expandByPoint(new THREE.Vector3(p[0], -p[1], -p[2])));
    const size = bbox.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.8;

    cameraVizCamera.position.set(
        center.x + dist * 0.6,
        center.y + dist * 0.8,
        center.z + dist * 0.6
    );
    cameraVizCamera.lookAt(center);
    cameraVizControls.update();
}

function syncToVideoTime() {
    if (!cameraVizData || !headcamVideo) return;

    const currentTime = headcamVideo.currentTime;
    const timestamps = cameraVizData.timestamps;

    // Find how many cameras should be visible at this video time
    let targetCount = 0;
    for (let i = 0; i < timestamps.length; i++) {
        if (currentTime >= timestamps[i]) {
            targetCount = i + 1;
        } else {
            break;
        }
    }
    targetCount = Math.min(targetCount, cameraFrustums.length);

    // If video looped (targetCount decreased), reset all frustums
    if (targetCount < lastSyncedIdx) {
        cameraFrustums.forEach(f => {
            f.visible = false;
            f.scale.set(1, 1, 1);
        });
        lastSyncedIdx = -1;
    }

    // Show/hide frustums based on current time
    for (let i = 0; i < cameraFrustums.length; i++) {
        const f = cameraFrustums[i];
        if (i < targetCount) {
            f.visible = true;
            // Pop-in animation for recently revealed
            const age = currentTime - timestamps[i];
            const t = Math.min(age / 0.15, 1);
            const s = 1 - Math.pow(1 - t, 3);
            f.scale.set(s, s, s);
        } else {
            f.visible = false;
        }
    }

    // Update depth map
    const depthIdx = targetCount - 1;
    if (depthIdx >= 0 && depthIdx !== lastSyncedIdx && depthDisplay) {
        lastSyncedIdx = depthIdx;
        depthDisplay.src = `static/data/headcam_cameras/depths/depth_${String(depthIdx).padStart(2, '0')}.jpg`;
    }
}

function animateCameraViz() {
    requestAnimationFrame(animateCameraViz);

    if (cameraVizControls) cameraVizControls.update();
    syncToVideoTime();

    if (cameraVizRenderer && cameraVizScene && cameraVizCamera) {
        cameraVizRenderer.render(cameraVizScene, cameraVizCamera);
    }
}
