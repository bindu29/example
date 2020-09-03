# Copy property file
COPY ./BirdPropertiesold.properties ./BirdProperties.properties
COPY ./datahandlerpropertiesold.properties ./datahandlerproperties.properties
COPY ./dataschedulerold.properties ./datascheduler.properties
RUN docker-compose up -d
