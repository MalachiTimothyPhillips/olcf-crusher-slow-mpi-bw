// Adapted from https://stackoverflow.com/a/53456430

def createStage(String name, String workDir, List stepList) {
  return {
    stage(name) {
      dir(workDir) {
        for (s in stepList) {
          catchError(buildResult: 'FAILURE', stageResult: 'FAILURE'){ sh s }
        }
      }
    }
  }
}


//def ethierStage = { ->
//  stage("ethier") {
//    List steps =  [
//      "echo 'I am okay'",
//      "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 1 1",
//      "echo 'I am also okay'"
//    ]
//    for (s in steps) {
//      catchError(buildResult: 'FAILURE', stageResult: 'FAILURE'){ sh s }
//    }
//    //catchError { sh "echo 'I am okay'" }
//    //catchError { sh "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 1 1" }
//    //catchError { sh "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 2 2" }
//    //catchError { sh "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 2 3" }
//    //catchError { sh "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 2 4" }
//    //catchError { sh "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 2 5" }
//    //catchError { sh "cd $NEKRS_EXAMPLES/ethier && nrsmpi ethier 2 6" }
//    //catchError { sh "echo 'I am also okay'" }
//  }
//}


def lowMachStage = { ->
  stage("lowMach") {
    sh "cd $NEKRS_EXAMPLES/lowMach && nrsmpi lowMach 2 1"
      sh "cd $NEKRS_EXAMPLES/lowMach && nrsmpi lowMach 2 1"
  }
}

def mvCylStage = { ->
  stage("mv_cyl") {
    sh "cd $NEKRS_EXAMPLES/mv_cyl && nrsmpi mv_cyl 2 1"
      sh "cd $NEKRS_EXAMPLES/mv_cyl && nrsmpi mv_cyl 2 2"
  }
}

def conjHtStage = { ->
  stage("conj_ht") {
    sh "cd $NEKRS_EXAMPLES/conj_ht && nrsmpi conj_ht 2 1"
  }
}

def channelStressStage = { ->
  stage("channelStress") {
    sh "cd $NEKRS_EXAMPLES/channel && nrsmpi channel 2 1"
  }
}


// "bigmem" runs on compute001
node("bigmem") {
  withEnv([
      'LD_LIBRARY_PATH=/soft/apps/packages/gcc/gcc-6.2.0/lib64:/soft/apps/packages/climate/mpich/3.2/gcc-6.2.0/lib',
      'LIBRARY_PATH=/usr/lib/x86_64-linux-gnu',
      "PATH=${pwd()}/install/bin:/soft/apps/packages/gcc/gcc-6.2.0/bin:/soft/apps/packages/climate/mpich/3.2/gcc-6.2.0/bin:/soft/apps/packages/cmake-3.14.3/bin:/soft/apps/packages/git-2.10.1/bin:/usr/lib/lightdm/lightdm:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/mcs/bin:/usr/local/bin:/software/common/bin:/soft/apps/bin:/soft/gnu/bin:/soft/com/bin:/soft/adm/bin:/homes/rahaman/bin/linux-Ubuntu_14.04-x86_64:/homes/rahaman/bin",
      "NEKRS_INSTALL_DIR=${pwd()}/install",
      "NEKRS_HOME=${pwd()}/install",
      "NEKRS_EXAMPLES=${pwd()}/install/examples",
      'OCCA_CUDA_ENABLED=0',
      'OCCA_HIP_ENABLED=0',
      'OCCA_OPENCL_ENABLED=0',
      'NEKRS_OCCA_MODE_DEFAULT=SERIAL',
      'NEKRS_CI=1'
  ]) {


    stage("Clone") {
      checkout scm
    }

    //stage ("Install") {
    //  sh 'env | sort'
    //  sh './nrsconfig'
    //  sh 'cmake --build build --target install -j 4'
    //}

    def ethierStage = createStage(
      "ethier", "${env.NEKRS_EXAMPLES}/ethier",
      [
        "pwd",
        "echo 'I am okay'",
        "nrsmpi ethier 1 1",
        "echo 'I am also okay'"
      ]
    )

    Map testStages = [ 
      "ethier" : ethierStage, 
      "lowMach": lowMachStage,
      "mv_cyl": mvCylStage,
      "conj_ht": conjHtStage,
      "channelStress": channelStressStage
    ]

    parallel(testStages)
  }
}
