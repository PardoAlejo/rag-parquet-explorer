#!/bin/bash
# Test runner script for RAG system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 RAG System Test Runner"
echo "=========================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest not found. Installing...${NC}"
    pip install pytest pytest-cov pytest-mock
fi

# Parse command line arguments
TEST_TYPE=${1:-all}

case $TEST_TYPE in
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        pytest -v
        ;;

    unit)
        echo -e "${GREEN}Running unit tests only...${NC}"
        pytest -v -m unit
        ;;

    integration)
        echo -e "${GREEN}Running integration tests...${NC}"
        pytest -v -m integration
        ;;

    fast)
        echo -e "${GREEN}Running fast tests (skip slow and external dependencies)...${NC}"
        pytest -v -m "not slow and not requires_ollama and not requires_models"
        ;;

    coverage)
        echo -e "${GREEN}Running tests with coverage report...${NC}"
        pytest --cov=rag --cov-report=html --cov-report=term
        echo -e "${YELLOW}📊 Coverage report generated in htmlcov/index.html${NC}"
        ;;

    embeddings)
        echo -e "${GREEN}Running embeddings tests...${NC}"
        pytest -v tests/test_embeddings.py
        ;;

    retrieval)
        echo -e "${GREEN}Running retrieval tests...${NC}"
        pytest -v tests/test_retrieval.py
        ;;

    llm)
        echo -e "${GREEN}Running LLM tests...${NC}"
        pytest -v tests/test_llm.py
        ;;

    utils)
        echo -e "${GREEN}Running utils tests...${NC}"
        pytest -v tests/test_utils.py
        ;;

    watch)
        echo -e "${GREEN}Running tests in watch mode (requires pytest-watch)...${NC}"
        if ! command -v ptw &> /dev/null; then
            echo -e "${YELLOW}Installing pytest-watch...${NC}"
            pip install pytest-watch
        fi
        ptw
        ;;

    help)
        echo "Usage: ./run_tests.sh [TYPE]"
        echo ""
        echo "Types:"
        echo "  all          - Run all tests (default)"
        echo "  unit         - Run unit tests only"
        echo "  integration  - Run integration tests only"
        echo "  fast         - Run fast tests (skip slow and external)"
        echo "  coverage     - Run with coverage report"
        echo "  embeddings   - Run embeddings module tests"
        echo "  retrieval    - Run retrieval module tests"
        echo "  llm          - Run LLM module tests"
        echo "  utils        - Run utils module tests"
        echo "  watch        - Run in watch mode (auto-rerun on changes)"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh              # Run all tests"
        echo "  ./run_tests.sh unit         # Run unit tests"
        echo "  ./run_tests.sh coverage     # Generate coverage report"
        exit 0
        ;;

    *)
        echo -e "${RED}❌ Unknown test type: $TEST_TYPE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Tests completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
