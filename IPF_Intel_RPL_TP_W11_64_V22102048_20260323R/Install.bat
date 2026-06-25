@echo off

%systemdrive%
cd %windir%/system32/
echo Please wait while installing drivers. Do not turn off or unplug the computer power during the installation...
Start /w pnputil.exe /add-driver "%~dp0\*.inf" /subdirs /install

exit