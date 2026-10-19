# Targets that run the Python helpers, after: pip install -r requirements.txt
# Pick the interpreter with -DPython3_EXECUTABLE=<path>. A target reruns only when an output is missing or its script changed.
#   cmake --build build --target data           download and tokenize Tiny Shakespeare
#   cmake --build build --target grad_goldens   PyTorch reference gradients for gpt_test

find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_Interpreter_FOUND)
    message(STATUS "Python 3 not found, the data and grad_goldens targets are unavailable")
    return()
endif()

set(DATA_DIR ${PROJECT_SOURCE_DIR}/data)
set(TOOLS_DIR ${PROJECT_SOURCE_DIR}/tools)

# Tiny Shakespeare
add_custom_command(
    OUTPUT ${DATA_DIR}/tiny_shakespeare.txt
    COMMAND ${Python3_EXECUTABLE} ${DATA_DIR}/tiny_shakespeare.py
    DEPENDS ${DATA_DIR}/tiny_shakespeare.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Downloading Tiny Shakespeare")

add_custom_command(
    OUTPUT ${DATA_DIR}/tiny_shakespeare.bin
    COMMAND ${Python3_EXECUTABLE} ${DATA_DIR}/tokenizer.py encode-file ${DATA_DIR}/tiny_shakespeare.txt ${DATA_DIR}/tiny_shakespeare.bin
    DEPENDS ${DATA_DIR}/tiny_shakespeare.txt ${DATA_DIR}/tokenizer.py
    COMMENT "Tokenizing Tiny Shakespeare")

add_custom_target(data DEPENDS ${DATA_DIR}/tiny_shakespeare.bin)

# Gradient goldens
set(GRAD_GOLDENS
    ${DATA_DIR}/gpt_dummy_tokens.bin
    ${DATA_DIR}/gpt_dummy_weights.bin
    ${DATA_DIR}/gpt_dummy_ref_gradients.bin)

add_custom_command(
    OUTPUT ${GRAD_GOLDENS}
    COMMAND ${Python3_EXECUTABLE} ${TOOLS_DIR}/prep_grad.py
    DEPENDS ${TOOLS_DIR}/prep_grad.py ${TOOLS_DIR}/gpt_reference.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generating gradient goldens with PyTorch")

add_custom_target(grad_goldens DEPENDS ${GRAD_GOLDENS})
