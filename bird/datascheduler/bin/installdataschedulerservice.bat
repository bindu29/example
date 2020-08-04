set SERVICE_NAME=CIDataSchedulerService
set "JVM=%JRE_HOME%\bin\server\jvm.dll"
if exist "%JVM%" goto foundJvm
rem Try to use the client jvm
set "JVM=%JRE_HOME%\bin\client\jvm.dll"
if exist "%JVM%" goto foundJvm
echo Warning: Neither 'server' nor 'client' jvm.dll was found at JRE_HOME.
set JVM=auto
:foundJvm
echo Using JVM:              "%JVM%"
set "BIRDSERVICE_HOME=%cd%"
set DISPLAYNAME=CIDataSchedulerService
set "CLASSPATH=%BIRDSERVICE_HOME%\datascheduler.jar;%BIRDSERVICE_HOME%\importerlib\*"

set EXECUTABLE=%BIRDSERVICE_HOME%\datascheduler.exe

 
REM JVM configuration
set PR_JVMMS=256
set PR_JVMMX=1024
set PR_JVMSS=4000
set PR_JVMOPTIONS=-Duser.language=en;-Duser.region=US
"%EXECUTABLE%" //DS//%SERVICE_NAME% ^
REM Install service

"%EXECUTABLE%" //IS//%SERVICE_NAME% ^
    --Description "CI DataschedulerService" ^
    --DisplayName "%DISPLAYNAME%" ^
    --Install %BIRDSERVICE_HOME%\datascheduler.exe ^
    --LogPath "%BIRDSERVICE_HOME%\logs" ^
    --StdOutput auto ^
    --StdError auto ^
    --Classpath "%CLASSPATH%" ^
    --Jvm "%JVM%" ^
    --StartMode jvm ^
    --StopMode jvm ^
    --StartClass "com.main.ds.DataschedulerService" ^
    --StopClass "com.main.ds.DataschedulerService" ^
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