// TypeScript wrapper for ProcLog WASM bindings
import {
  parse_program as wasmParseProgram,
  evaluate_program as wasmEvaluateProgram,
  run_tests as wasmRunTests,
} from '../wasm/proclog_wasm.js';

// Type definitions for ProcLog structures
export interface Program {
  statements: Statement[];
}

export interface Statement {
  // Will be refined based on actual AST structure
  [key: string]: any;
}

export interface AnswerSet {
  atoms: Atom[];
}

export interface Atom {
  predicate: string;
  terms: Term[];
}

export interface Term {
  [key: string]: any;
}

export interface EvaluationResult {
  answerSets: AnswerSet[];
}

export interface TestResult {
  testName: string;
  passed: boolean;
  totalCases: number;
  passedCases: number;
  caseResults: TestCaseResult[];
}

export interface TestCaseResult {
  passed: boolean;
  message: string;
}

export interface TestRunResult {
  results: TestResult[];
  totalTests: number;
  passedTests: number;
  failedTests: number;
}

/**
 * Parse a ProcLog program from source code.
 * @param source - The ProcLog source code to parse
 * @returns Parsed program structure
 * @throws Error if parsing fails
 */
export function parseProgram(source: string): Program {
  return wasmParseProgram(source);
}

/**
 * Evaluate a ProcLog program and return answer sets.
 * @param source - The ProcLog source code to evaluate
 * @returns Evaluation result with answer sets
 * @throws Error if evaluation fails
 */
export function evaluateProgram(source: string): EvaluationResult {
  return wasmEvaluateProgram(source);
}

/**
 * Run test blocks in a ProcLog program.
 * @param source - The ProcLog source code with test blocks
 * @param useSatSolver - Whether to use the SAT solver backend (default: false)
 * @returns Test run results
 * @throws Error if test execution fails
 */
export function runTests(source: string, useSatSolver: boolean = false): TestRunResult {
  return wasmRunTests(source, useSatSolver);
}
