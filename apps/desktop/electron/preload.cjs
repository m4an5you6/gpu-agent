const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('gpucloudDesktop', {
  getConnection: profile => ipcRenderer.invoke('gpucloud:connection', profile),
  revalidateConnection: () => ipcRenderer.invoke('gpucloud:connection:revalidate'),
  touchBackend: profile => ipcRenderer.invoke('gpucloud:backend:touch', profile),
  getGatewayWsUrl: profile => ipcRenderer.invoke('gpucloud:gateway:ws-url', profile),
  getBootProgress: () => ipcRenderer.invoke('gpucloud:boot-progress:get'),
  getConnectionConfig: profile => ipcRenderer.invoke('gpucloud:connection-config:get', profile),
  saveConnectionConfig: payload => ipcRenderer.invoke('gpucloud:connection-config:save', payload),
  applyConnectionConfig: payload => ipcRenderer.invoke('gpucloud:connection-config:apply', payload),
  testConnectionConfig: payload => ipcRenderer.invoke('gpucloud:connection-config:test', payload),
  probeConnectionConfig: remoteUrl => ipcRenderer.invoke('gpucloud:connection-config:probe', remoteUrl),
  oauthLoginConnectionConfig: remoteUrl => ipcRenderer.invoke('gpucloud:connection-config:oauth-login', remoteUrl),
  oauthLogoutConnectionConfig: remoteUrl => ipcRenderer.invoke('gpucloud:connection-config:oauth-logout', remoteUrl),
  profile: {
    get: () => ipcRenderer.invoke('gpucloud:profile:get'),
    set: name => ipcRenderer.invoke('gpucloud:profile:set', name)
  },
  api: request => ipcRenderer.invoke('gpucloud:api', request),
  notify: payload => ipcRenderer.invoke('gpucloud:notify', payload),
  requestMicrophoneAccess: () => ipcRenderer.invoke('gpucloud:requestMicrophoneAccess'),
  readFileDataUrl: filePath => ipcRenderer.invoke('gpucloud:readFileDataUrl', filePath),
  readFileText: filePath => ipcRenderer.invoke('gpucloud:readFileText', filePath),
  selectPaths: options => ipcRenderer.invoke('gpucloud:selectPaths', options),
  writeClipboard: text => ipcRenderer.invoke('gpucloud:writeClipboard', text),
  saveImageFromUrl: url => ipcRenderer.invoke('gpucloud:saveImageFromUrl', url),
  saveImageBuffer: (data, ext) => ipcRenderer.invoke('gpucloud:saveImageBuffer', { data, ext }),
  saveClipboardImage: () => ipcRenderer.invoke('gpucloud:saveClipboardImage'),
  getPathForFile: file => {
    try {
      return webUtils.getPathForFile(file) || ''
    } catch {
      return ''
    }
  },
  normalizePreviewTarget: (target, baseDir) => ipcRenderer.invoke('gpucloud:normalizePreviewTarget', target, baseDir),
  watchPreviewFile: url => ipcRenderer.invoke('gpucloud:watchPreviewFile', url),
  stopPreviewFileWatch: id => ipcRenderer.invoke('gpucloud:stopPreviewFileWatch', id),
  setTitleBarTheme: payload => ipcRenderer.send('gpucloud:titlebar-theme', payload),
  setPreviewShortcutActive: active => ipcRenderer.send('gpucloud:previewShortcutActive', Boolean(active)),
  openExternal: url => ipcRenderer.invoke('gpucloud:openExternal', url),
  fetchLinkTitle: url => ipcRenderer.invoke('gpucloud:fetchLinkTitle', url),
  settings: {
    getDefaultProjectDir: () => ipcRenderer.invoke('gpucloud:setting:defaultProjectDir:get'),
    setDefaultProjectDir: dir => ipcRenderer.invoke('gpucloud:setting:defaultProjectDir:set', dir),
    pickDefaultProjectDir: () => ipcRenderer.invoke('gpucloud:setting:defaultProjectDir:pick')
  },
  revealLogs: () => ipcRenderer.invoke('gpucloud:logs:reveal'),
  getRecentLogs: () => ipcRenderer.invoke('gpucloud:logs:recent'),
  readDir: dirPath => ipcRenderer.invoke('gpucloud:fs:readDir', dirPath),
  gitRoot: startPath => ipcRenderer.invoke('gpucloud:fs:gitRoot', startPath),
  terminal: {
    dispose: id => ipcRenderer.invoke('gpucloud:terminal:dispose', id),
    resize: (id, size) => ipcRenderer.invoke('gpucloud:terminal:resize', id, size),
    start: options => ipcRenderer.invoke('gpucloud:terminal:start', options),
    write: (id, data) => ipcRenderer.invoke('gpucloud:terminal:write', id, data),
    onData: (id, callback) => {
      const channel = `gpucloud:terminal:${id}:data`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    },
    onExit: (id, callback) => {
      const channel = `gpucloud:terminal:${id}:exit`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    }
  },
  onClosePreviewRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('gpucloud:close-preview-requested', listener)
    return () => ipcRenderer.removeListener('gpucloud:close-preview-requested', listener)
  },
  onOpenUpdatesRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('gpucloud:open-updates', listener)
    return () => ipcRenderer.removeListener('gpucloud:open-updates', listener)
  },
  onWindowStateChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('gpucloud:window-state-changed', listener)
    return () => ipcRenderer.removeListener('gpucloud:window-state-changed', listener)
  },
  onPreviewFileChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('gpucloud:preview-file-changed', listener)
    return () => ipcRenderer.removeListener('gpucloud:preview-file-changed', listener)
  },
  onBackendExit: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('gpucloud:backend-exit', listener)
    return () => ipcRenderer.removeListener('gpucloud:backend-exit', listener)
  },
  onPowerResume: callback => {
    const listener = () => callback()
    ipcRenderer.on('gpucloud:power-resume', listener)
    return () => ipcRenderer.removeListener('gpucloud:power-resume', listener)
  },
  onBootProgress: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('gpucloud:boot-progress', listener)
    return () => ipcRenderer.removeListener('gpucloud:boot-progress', listener)
  },
  // First-launch bootstrap progress -- emitted by the install.ps1 stage
  // runner in main.cjs (apps/desktop/electron/bootstrap-runner.cjs).
  // Renderer's install overlay subscribes to live events and queries the
  // current snapshot via getBootstrapState() to recover after a devtools
  // reload mid-bootstrap.
  getBootstrapState: () => ipcRenderer.invoke('gpucloud:bootstrap:get'),
  resetBootstrap: () => ipcRenderer.invoke('gpucloud:bootstrap:reset'),
  repairBootstrap: () => ipcRenderer.invoke('gpucloud:bootstrap:repair'),
  cancelBootstrap: () => ipcRenderer.invoke('gpucloud:bootstrap:cancel'),
  onBootstrapEvent: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('gpucloud:bootstrap:event', listener)
    return () => ipcRenderer.removeListener('gpucloud:bootstrap:event', listener)
  },
  getVersion: () => ipcRenderer.invoke('gpucloud:version'),
  uninstall: {
    summary: () => ipcRenderer.invoke('gpucloud:uninstall:summary'),
    run: mode => ipcRenderer.invoke('gpucloud:uninstall:run', { mode })
  },
  updates: {
    check: () => ipcRenderer.invoke('gpucloud:updates:check'),
    apply: opts => ipcRenderer.invoke('gpucloud:updates:apply', opts),
    getBranch: () => ipcRenderer.invoke('gpucloud:updates:branch:get'),
    setBranch: name => ipcRenderer.invoke('gpucloud:updates:branch:set', name),
    onProgress: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('gpucloud:updates:progress', listener)
      return () => ipcRenderer.removeListener('gpucloud:updates:progress', listener)
    }
  }
})
