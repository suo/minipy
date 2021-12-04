add_library(parser
    error_report.cpp
    lexer.cpp
    parser.cpp
    source_range.cpp
    strtod.cpp
)

target_link_libraries(parser common)
