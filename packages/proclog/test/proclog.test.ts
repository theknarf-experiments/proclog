import { describe, it, expect } from 'vitest';
import { parseProgram, evaluateProgram, runTests } from '../src/index.js';

describe('ProcLog Parser', () => {
  it('should parse simple facts', () => {
    const source = 'parent(alice, bob).';
    const result = parseProgram(source);

    expect(result).toBeDefined();
    expect(result.statements).toBeDefined();
    expect(result.statements.length).toBeGreaterThan(0);
  });

  it('should parse multiple facts', () => {
    const source = `
      parent(alice, bob).
      parent(bob, charlie).
    `;
    const result = parseProgram(source);

    expect(result.statements.length).toBe(2);
  });

  it('should parse rules', () => {
    const source = `
      parent(alice, bob).
      ancestor(X, Y) :- parent(X, Y).
    `;
    const result = parseProgram(source);

    expect(result.statements.length).toBe(2);
  });

  it('should throw error on invalid syntax', () => {
    const source = 'invalid syntax here!!!';

    expect(() => parseProgram(source)).toThrow();
  });
});

describe('ProcLog Evaluation', () => {
  it('should evaluate simple facts', () => {
    const source = 'parent(alice, bob).';
    const result = evaluateProgram(source);

    expect(result).toBeDefined();
    expect(result.answerSets).toBeDefined();
    expect(result.answerSets.length).toBeGreaterThan(0);
  });

  it('should evaluate rules and derive facts', () => {
    const source = `
      parent(alice, bob).
      parent(bob, charlie).
      ancestor(X, Y) :- parent(X, Y).
      ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
    `;
    const result = evaluateProgram(source);

    expect(result.answerSets.length).toBeGreaterThan(0);
    // Should have parent facts and derived ancestor facts
    expect(result.answerSets[0].atoms.length).toBeGreaterThan(2);
  });

  it('should handle empty program', () => {
    const source = '';
    const result = evaluateProgram(source);

    expect(result.answerSets).toBeDefined();
  });
});

describe('ProcLog Test Runner', () => {
  it('should run test blocks', () => {
    const source = `
      parent(alice, bob).

      #test "parent facts" {
        ?- parent(alice, bob).
      }
    `;
    const result = runTests(source);

    expect(result).toBeDefined();
    expect(result.results).toBeDefined();
    expect(result.results.length).toBe(1);
    expect(result.results[0].passed).toBe(true);
  });

  it('should report test failures', () => {
    const source = `
      parent(alice, bob).

      #test "failing test" {
        ?- parent(bob, alice).
        + parent(bob, alice).
      }
    `;
    const result = runTests(source);

    expect(result.results.length).toBe(1);
    expect(result.results[0].passed).toBe(false);
  });

  it('should run multiple test blocks', () => {
    const source = `
      parent(alice, bob).
      parent(bob, charlie).

      #test "first test" {
        ?- parent(alice, bob).
      }

      #test "second test" {
        ?- parent(bob, charlie).
      }
    `;
    const result = runTests(source);

    expect(result.results.length).toBe(2);
    expect(result.results[0].passed).toBe(true);
    expect(result.results[1].passed).toBe(true);
  });

  it('should provide test summary', () => {
    const source = `
      parent(alice, bob).

      #test "passing test" {
        ?- parent(alice, bob).
      }
    `;
    const result = runTests(source);

    expect(result.totalTests).toBe(1);
    expect(result.passedTests).toBe(1);
    expect(result.failedTests).toBe(0);
  });
});
