@echo off

if DEFINED JAVA_HOME goto cont

:err
ECHO JAVA_HOME environment variable must be set! 1>&2
                                                                                

                                                                
EXIT /B 1 
:cont
java -jar datahandler.jar
