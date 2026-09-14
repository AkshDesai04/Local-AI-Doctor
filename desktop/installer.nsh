; Local AI Doctor does not ship an auto-updater. Electron Builder normally
; stores a second full installer in LOCALAPPDATA for updater handoff, which
; would waste roughly another 2 GiB for the CUDA build. Delete only that exact
; generated cache file after the application files have been installed.
!macro customInstall
  Delete "$LOCALAPPDATA\${APP_INSTALLER_STORE_FILE}"
!macroend
