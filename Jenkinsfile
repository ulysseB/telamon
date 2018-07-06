pipeline {
    agent { label 'ficus' }

    stages {
        stage('Build') {
            steps {
                sh 'cargo build'
            }
        }
    }
}
