//! インクリメンタルに自動部分ライブラリを作っていく
//! 計算グラフは最初はf64飲みで作っていく
//! 最初は二つのスカラの足し算のみを作っていく

/// 二つのスカラを足し合わせることを表すノード
pub struct Add {
    pub x: f64,
    pub y: f64,
    pub output: f64,
}

impl Add {
    fn forward(&mut self) {
        self.output = self.x + self.y;
    }
}

pub struct Graph {
    pub add: Add,
}

impl Graph {
    fn new() -> Self {
        Self {
            add: Add {
                x: 0.0,
                y: 0.0,
                output: 0.0,
            },
        }
    }

    fn forward(&mut self, x: f64, y: f64) -> f64 {
        self.add.x = x;
        self.add.y = y;
        self.add.forward();
        self.add.output
    }
}
