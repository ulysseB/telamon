pipeline {
  agent { label 'ficus' }

  stages {
    stage('build') {
      steps {
        dir('kernels') {
          sh 'cargo build --features=cuda --bench=cuda-search'
          sh 'cargo build --features=cuda --bench=cuda-bound'
          sh 'cargo build --features=cuda --bench=cuda-deadend'
        }
      }
    } 
    stage('bench') {
      steps {
        sh 'mkdir -p output'
        dir('kernels') {
          sh 'cargo bench --bench=cuda-deadend > ../output/cuda_deadend'
          sh 'cargo bench --bench=cuda-bound > ../output/cuda_bound'
          sh 'RUST_LOG=telamon::explorer=warn cargo bench --bench=cuda-search > ../output/cuda_search 2>../output/cuda_search_log'
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
