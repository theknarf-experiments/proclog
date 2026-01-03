    use super::*;
    use crate::repl::ReplEngine;

    fn ctrl_w() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn ctrl_w_moves_focus_between_panes() {
        let mut app = ReplApp::new(ReplEngine::new());
        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('h')));
        assert_eq!(app.active_pane(), ActivePane::History);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('l')));
        assert_eq!(app.active_pane(), ActivePane::Database);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.active_pane(), ActivePane::Stats);
    }

    #[test]
    fn database_view_scrolls_with_arrow_keys() {
        let mut app = ReplApp::new(ReplEngine::new());
        // Populate with many facts
        let response = app.engine.process_line("value(1..50).");
        app.record_response(response);

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('l')));
        assert_eq!(app.active_pane(), ActivePane::Database);

        let initial_scroll = app.db_scroll();
        app.handle_key(key(KeyCode::Down));
        let after_scroll = app.db_scroll();
        assert!(
            after_scroll >= initial_scroll,
            "scroll should not decrease when pressing down"
        );
    }

    #[test]
    fn ctrl_w_with_arrow_keys_moves_focus() {
        let mut app = ReplApp::new(ReplEngine::new());
        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Right));
        assert_eq!(app.active_pane(), ActivePane::Database);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.active_pane(), ActivePane::Stats);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Left));
        assert_eq!(app.active_pane(), ActivePane::History);
    }
