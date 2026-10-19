# Checking and profiling targets
#   cmake --build build --config Release --target sanitize   gpt_test under every compute-sanitizer tool
#   cmake --build build --config Release --target nsys       GPU time per NVTX range and per kernel for a few training steps

# compute-sanitizer: runs the gradient test (tiny model) under each tool, stopping at the first one that reports a problem.
find_program(COMPUTE_SANITIZER_EXECUTABLE compute-sanitizer HINTS "${CUDAToolkit_BIN_DIR}/../compute-sanitizer")

if(COMPUTE_SANITIZER_EXECUTABLE)
    set(SANITIZE_COMMANDS)

    # -q prints only findings. Without it, the "ERROR SUMMARY: 0 errors" line looks like a compiler error to MSBuild and fails the target.
    foreach(tool memcheck racecheck initcheck synccheck)
        list(APPEND SANITIZE_COMMANDS COMMAND ${COMPUTE_SANITIZER_EXECUTABLE} -q --tool ${tool} --error-exitcode 1 $<TARGET_FILE:gpt_test>)
    endforeach()

    add_custom_target(sanitize
        ${SANITIZE_COMMANDS}
        DEPENDS gpt_test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        USES_TERMINAL
        COMMENT "Running gpt_test under compute-sanitizer (memcheck, racecheck, initcheck, synccheck)")
else()
    message(STATUS "compute-sanitizer not found, the sanitize target is unavailable")
endif()

# Nsight Systems: records 3 training steps into build/profiles/train.nsys-rep and prints the summaries.
file(GLOB NSYS_HINTS "C:/Program Files/NVIDIA Corporation/Nsight Systems */target-windows-x64")
find_program(NSYS_EXECUTABLE nsys HINTS ${NSYS_HINTS})

if(NSYS_EXECUTABLE)
    set(PROFILES_DIR ${CMAKE_BINARY_DIR}/profiles)

    add_custom_target(nsys
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROFILES_DIR}
        COMMAND ${NSYS_EXECUTABLE} profile --trace=cuda,nvtx --cuda-event-trace=false --sample=none --cpuctxsw=none --force-overwrite=true
                --output=${PROFILES_DIR}/train
                $<TARGET_FILE:gpt_train> --epochs 1 --max-steps 3 --batch-size 8 --batch-accum-steps 4
                --data ${PROJECT_SOURCE_DIR}/data/tiny_shakespeare.bin
        COMMAND ${NSYS_EXECUTABLE} stats --force-export=true --report nvtx_gpu_proj_sum --report cuda_gpu_kern_sum
                ${PROFILES_DIR}/train.nsys-rep
        DEPENDS gpt_train
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        USES_TERMINAL
        COMMENT "Profiling 3 training steps with Nsight Systems")
else()
    message(STATUS "nsys not found, the nsys target is unavailable")
endif()
