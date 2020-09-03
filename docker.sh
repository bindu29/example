#copy property files
#sudo cp ./BirdPropertiesold.properties ./BirdProperties.properties
#sudo cp ./datahandlerpropertiesold.properties ./datahandlerproperties.properties
#sudo cp ./dataschedulerold.properties ./datascheduler.properties

# copy ip addresses of components 
read -p "Please Enter IP Address for Tomcat: " tomip
read -p "Please Enter IP Address for PostgreSQL: " postip
read -p "Please Enter IP Address for RabbitMQ: " rabbitip
read -p "Please Enter IP Address for Columnar Storage: " columnarip
read -p "Please Enter IP Address for Python: " pythonip
read -p "Please Enter IP Address for Datahandler: " dhip

#paste ip into properities

sed -i -e "s+postip+$postip+g" BirdProperties.properties datahandlerproperties.properties datascheduler.properties
sed -i -e "s+columnip+$columnarip+g" BirdProperties.properties datahandlerproperties.properties datascheduler.properties
sed -i -e "s+rabbitip+$rabbitip+g" BirdProperties.properties datahandlerproperties.properties datascheduler.properties
sed -i -e "s+tomip+$tomip+g" BirdProperties.properties datahandlerproperties.properties datascheduler.properties
sed -i -e "s+pythonip+$pythonip+g" BirdProperties.properties datahandlerproperties.properties datascheduler.properties
sed -i -e "s+dhip+$dhip+g" BirdProperties.properties datahandlerproperties.properties datascheduler.properties


#copy build to containers
sudo docker cp Bird.war tomcatbird:/usr/local/tomcat/webapps/
sudo docker cp datahandler birdhandler:/opt
sudo docker cp datascheduler birdhandler:/opt
sudo docker cp config.xml birdstore:/etc/clickhouse-server
sudo docker cp users.xml birdstore:/etc/clickhouse-server

# copy properities
sudo docker cp BirdProperties.properties tomcatbird:/usr/local/tomcat/webapps/Bird/WEB-INF/BirdProperties.properties
sudo docker cp datahandlerproperties.properties birdhandler:/opt/datahandler/bin/config/datahandlerproperties.properties
sudo docker cp datascheduler.properties birdhandler:/opt/datascheduler/bin/config/datascheduler.properties


# start and stop tomcat and handler

sudo docker stop tomcatbird && docker start tomcatbird
sudo docker stop birdhandler && docker start birdhandler
