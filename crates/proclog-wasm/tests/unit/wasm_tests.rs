    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_parse_program() {
        let result = parse_program("parent(alice, bob).");
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_evaluate_program() {
        let result = evaluate_program("parent(alice, bob).");
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_run_tests() {
        let source = r#"
            parent(alice, bob).

            #test "simple test" {
                ?- parent(alice, bob).
            }
        "#;
        let result = run_tests(source, false);
        assert!(result.is_ok());
    }
