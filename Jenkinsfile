pipeline {
  agent { label 'ficus' }

  stages {
    stage('build') {
      steps {
        sh 'cargo build --features=cuda --release'
      }
    } 
    stage('bench') {
      steps {
        sh 'mkdir -p output'
        dir('kernels') {
          sh 'cargo bench --features=cuda --bench=cuda-deadend > ../output/cuda_deadend'
          sh 'cargo bench --features=cuda --bench=cuda-bound > ../output/cuda_bound'
          sh 'RUST_LOG=telamon::explorer=warn cargo bench --features=cuda --bench=cuda-search > ../output/cuda_search'
        }
      }
      
      post {
        success {
          archiveArtifacts(artifacts: 'output/*')
        }
      }
    }
  }
}
