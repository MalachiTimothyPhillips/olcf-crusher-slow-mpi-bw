// Creates a stage for a given test case.  See usage in node() below.
def createTestStage(String name, String workDir, List stepList) {
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
      'TRAVIS=true'
  ]) {
    
    // =====================================================
    // Instantiating stages for test cases
    //   1. Create a stage for each test case
    //   2. Create a Map, `testStages` with the stages
    //   3. Use the Map to create parallel runs (below)
    // =====================================================

    def ethierStage = createTestStage(
      "ethier", 
      "${env.NEKRS_EXAMPLES}/ethier",
      [
        "nrsmpi ethier 1 1",
        "nrsmpi ethier 2 2",
        "nrsmpi ethier 2 3",
        "nrsmpi ethier 2 4",
        "nrsmpi ethier 2 5",
        "nrsmpi ethier 2 6"
      ]
    )

    def lowMachStage = createTestStage(
      "lowMach", 
      "${env.NEKRS_EXAMPLES}/lowMach",
      [
        "nrsmpi lowMach 2 1"
      ]
    )

    def mvCylStage = createTestStage(
      "mv_cyl", 
      "${env.NEKRS_EXAMPLES}/mv_cyl",
      [
        "nrsmpi mv_cyl 2 1",
      ]
    )

    def conjHtStage = createTestStage(
      "conj_ht", 
      "${env.NEKRS_EXAMPLES}/conj_ht",
      [ "nrsmpi conj_ht 2 1" ]
    )

    def channelStressStage = createTestStage(
      "channel", 
      "${env.NEKRS_EXAMPLES}/channel",
      [ "nrsmpi channel 2 1" ]
    )

    Map testStages = [ 
      "ethier" : ethierStage, 
      "lowMach": lowMachStage,
      "mv_cyl": mvCylStage,
      "conj_ht": conjHtStage,
      "channelStress": channelStressStage
    ]
    
    // =====================================================
    // Run all stages (including setup and tests)
    // =====================================================

    stage("Clone") {
      checkout scm
    }

    stage ("Install") {
      sh 'env | sort'
      sh './makenrs'
    }

    parallel(testStages)

  } // end withEnv
} // end node("bigmem")
