set SERVICE_NAME=CIDataHandlerService
set "JVM=%JRE_HOME%\bin\server\jvm.dll"
if exist "%JVM%" goto foundJvm
rem Try to use the client jvm
set "JVM=%JRE_HOME%\bin\client\jvm.dll"
if exist "%JVM%" goto foundJvm
echo Warning: Neither 'server' nor 'client' jvm.dll was found at JRE_HOME.
set JVM=auto
:foundJvm
echo Using JVM:              "%JVM%"
set "CI_DATASERVICE_HOME=%cd%"
set DISPLAYNAME=CIDataHandlerService
set "CLASSPATH=%CI_DATASERVICE_HOME%\datahandler.jar;%CI_DATASERVICE_HOME%\importerlib\*"

set EXECUTABLE=%CI_DATASERVICE_HOME%\birddatahandler.exe

 
REM JVM configuration
set PR_JVMMS=256
set PR_JVMMX=1024
set PR_JVMSS=4000
set PR_JVMOPTIONS=-Duser.language=en;-Duser.region=US
"%EXECUTABLE%" //DS//%SERVICE_NAME% ^
REM Install service

"%EXECUTABLE%" //IS//%SERVICE_NAME% ^
    --Description "CIDataHandler" ^
    --DisplayName "%DISPLAYNAME%" ^
    --Install %CI_DATASERVICE_HOME%\birddatahandler.exe ^
    --LogPath "%CI_DATASERVICE_HOME%\logs" ^
    --StdOutput auto ^
    --StdError auto ^
    --Classpath "%CLASSPATH%" ^
    --Jvm "%JVM%" ^
    --StartMode jvm ^
    --StopMode jvm ^
    --StartClass "com.main.dh.server.DHServerService" ^
    --StopClass "com.main.dh.server.DHServerService" ^
    --StartParams start ^
    --StopParams stop ^
    --JvmMs 256 ^
    --JvmMx 2048
	
if not errorlevel 1 goto installed
echo Failed installing '%SERVICE_NAME%' service
goto end
:installed
echo The service '%SERVICE_NAME%' has been installed.

:end
cd "%CURRENT_DIR%"