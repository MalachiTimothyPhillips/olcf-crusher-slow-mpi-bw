// Adapted from https://stackoverflow.com/a/53456430

def ethierStage = { ->
  stage("ethier") {
    sh 'echo ethier ${FOO}'
      sh 'which mpic++'
      sh 'which mpif90'
      sh 'sleep 5'
  }
}

def lowMachStage = { ->
  stage("lowMach") {
    sh 'echo lowMach ${FOO}'
      sh 'sleep 5'
  }
}

List testStages = [
  ["ethier" : ethierStage],
  ["lowMach": lowMachStage]
]

// "bigmem" runs on compute001
node("bigmem") {
  withEnv([
      'LD_LIBRARY_PATH=/soft/apps/packages/gcc/gcc-6.2.0/lib64:/soft/apps/packages/climate/mpich/3.2/gcc-6.2.0/lib',
      'LIBRARY_PATH=/usr/lib/x86_64-linux-gnu',
      "PATH=${pwd()}/install/bin:/soft/apps/packages/gcc/gcc-6.2.0/bin:/soft/apps/packages/climate/mpich/3.2/gcc-6.2.0/bin:/soft/apps/packages/cmake-3.14.3/bin:/soft/apps/packages/git-2.10.1/bin:/usr/lib/lightdm/lightdm:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/mcs/bin:/usr/local/bin:/software/common/bin:/soft/apps/bin:/soft/gnu/bin:/soft/com/bin:/soft/adm/bin:/homes/rahaman/bin/linux-Ubuntu_14.04-x86_64:/homes/rahaman/bin",
      "NEKRS_INSTALL_DIR=${pwd()}/install",
      "NEKRS_HOME=${pwd()}/install",
      "NEKRS_EXAMPLES=${pwd()}/install/examples",
      "OCCA_CACHE_DIR=${pwd()}/occa_cache",
      'OCCA_CUDA_ENABLED=0',
      'OCCA_HIP_ENABLED=0',
      'OCCA_OPENCL_ENABLED=0',
      'NEKRS_OCCA_MODE_DEFAULT=SERIAL',
      'NEKRS_CI=1'
  ]) {

    stage("Clone") {
      checkout([$class: 'GitSCM', branches: [[name: '*/master'], [name: '*/next']], extensions: [[$class: 'CloneOption', noTags: true, reference: '', shallow: true], [$class: 'WipeWorkspace']], userRemoteConfigs: [[url: 'https://github.com/Nek5000/nekRS.git']]])
    }

    stage ("Install") {
      sh 'env | sort'
      sh './nrsconfig'
      sh 'cmake --build build --target install -j 4'
    }

    stage ("Warm-up") {
      sh 'cd $NEKRS_EXAMPLES/ethier && nrspre ethier 1'
    }

    parallel {
      stage("ethier") {
          sh 'echo ethier ${FOO}'
          sh 'which mpic++'
          sh 'which mpif90'
          sh 'sleep 5'
      }

      stage("lowMach") {
        sh 'echo lowMach ${FOO}'
          sh 'sleep 5'
      }
    }

    //for (test in testStages) {
    //  parallel(test)
    //}
  }
}
