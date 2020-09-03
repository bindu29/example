pipeline {
environment {
registry = "birdsriyam/bird"
registryCredential = '5beee4f0-1427-415a-8373-e260ce3428f1'
dockerImage = ''
}
agent any
stages {
stage('Building BirdTomcat') {
steps{
customWorkspace '/opt/builds/docker/dockertomcat'
script {
dockerImage = docker.build registry + ":BirdTC$BUILD_NUMBER"
}
}
}
stage('push BirdTomcat') {
steps{
script {
docker.withRegistry( '', registryCredential ) {
dockerImage.push()
}
}
}
}
}
}
