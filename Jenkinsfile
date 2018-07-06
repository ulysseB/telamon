pipeline {
    agent { label 'ficus' }

    stages {
        stage('Build') {
            steps {
                cache(maxCacheSize: 2000, caches: [
                    [$class: 'ArbitraryFileCache', excludes: '', includes: '**/*', path: '${WORKSPACE}/target']
                ]) {
                    sh 'cargo build'
                }
            }
        }
    }
}
