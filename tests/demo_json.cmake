file(MAKE_DIRECTORY "${WORK}")
set(quoted_image "${WORK}/a\"quoted.png")
configure_file("${SOURCE}/00001.png" "${quoted_image}" COPYONLY)
execute_process(COMMAND "${TOOL}" --demo "${quoted_image}" "${SOURCE}/00002.png"
    WORKING_DIRECTORY "${WORK}" RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Demo failed: ${output}\n${error}")
endif()
execute_process(COMMAND "${CHECKER}" --check-demo-json
    "${WORK}/demo_output/metadata.json" "${quoted_image}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Demo metadata check failed: ${output}\n${error}")
endif()
file(REMOVE_RECURSE "${WORK}")
