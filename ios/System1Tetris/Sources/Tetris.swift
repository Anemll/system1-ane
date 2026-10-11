// Swift port of coreai/jeff_tetris.py: board, placements, El-Tetris oracle, option text. Keep in step with the Python.
import Foundation

enum Tetris {
    static let width = 10
    static let height = 20
    static let rules = "10 by 20 board. # is filled, a dot is empty. The top row is printed first. "
        + "Drop the named piece in one rotation and column. Do not leave a hole you can avoid."
    static let instructions = "Choose where to drop this piece."
    static let pieceNames = ["I", "O", "T", "S", "Z", "J", "L"]   // dict order of PIECES in Python

    typealias Cell = (r: Int, c: Int)
    typealias Board = [[UInt8]]

    private static let raw: [String: [Cell]] = [
        "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
        "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "T": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
        "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
        "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
    ]
    // El-Tetris / Dellacherie weights.
    private static let wLanding = -4.500158825082766, wEroded = 3.4181268101392694
    private static let wRowT = -3.2178882868487753, wColT = -9.348695305445199
    private static let wHoles = -7.899265427351652, wWells = -3.3855972247263626

    private static func normalize(_ cells: [Cell]) -> [Cell] {
        let mr = cells.map(\.r).min()!, mc = cells.map(\.c).min()!
        return cells.map { (r: $0.r - mr, c: $0.c - mc) }.sorted { ($0.r, $0.c) < ($1.r, $1.c) }
    }
    private static func rotations(_ cells: [Cell]) -> [[Cell]] {
        var seen: [[Cell]] = []
        var cur = cells
        for _ in 0..<4 {
            let n = normalize(cur)
            if !seen.contains(where: { a in a.count == n.count && zip(a, n).allSatisfy { $0 == $1 } }) { seen.append(n) }
            cur = cur.map { (r: $0.c, c: -$0.r) }
        }
        return seen
    }
    static let pieces: [String: [[Cell]]] = Dictionary(uniqueKeysWithValues: raw.map { ($0.key, rotations($0.value)) })

    static func emptyBoard() -> Board { Array(repeating: Array(repeating: 0, count: width), count: height) }
    static func boardText(_ b: Board) -> String {
        b.map { String($0.map { $0 != 0 ? "#" : "." }) }.joined(separator: "\n")
    }

    private static func fits(_ b: Board, _ shape: [Cell], _ row: Int, _ col: Int) -> Bool {
        for (dr, dc) in shape {
            let r = row + dr, c = col + dc
            if c < 0 || c >= width || r >= height { return false }
            if r >= 0 && b[r][c] != 0 { return false }
        }
        return true
    }
    static func dropRow(_ b: Board, _ shape: [Cell], _ col: Int) -> Int? {
        if !fits(b, shape, 0, col) { return nil }
        var row = 0
        while fits(b, shape, row + 1, col) { row += 1 }
        return row
    }
    private static func paint(_ b: Board, _ shape: [Cell], _ row: Int, _ col: Int) -> Board {
        var n = b
        for (dr, dc) in shape where row + dr >= 0 { n[row + dr][col + dc] = 1 }
        return n
    }
    private static func clear(_ b: Board, _ shape: [Cell], _ row: Int) -> (Board, Int, Int) {
        var kept: Board = [], cleared = Set<Int>()
        for (r, line) in b.enumerated() { if line.allSatisfy({ $0 != 0 }) { cleared.insert(r) } else { kept.append(line) } }
        while kept.count < height { kept.insert(Array(repeating: 0, count: width), at: 0) }
        let n = cleared.count
        let cells = shape.filter { cleared.contains(row + $0.r) }.count
        return (kept, cells * n, n)
    }
    private static func columnHeight(_ b: Board, _ col: Int) -> Int {
        for r in 0..<height where b[r][col] != 0 { return height - r }
        return 0
    }

    struct Placement {
        let key: String
        let rotation: Int, column: Int, row: Int
        let rating: Double
        let board: Board
        let lines: Int, holes: Int
        let shape: [Cell]
    }

    static func placements(_ b: Board, _ piece: String) -> [Placement] {
        var found: [Placement] = []
        for (rotation, shape) in pieces[piece]!.enumerated() {
            let w = shape.map(\.c).max()! + 1
            for col in 0...(width - w) {
                guard let row = dropRow(b, shape, col) else { continue }
                let painted = paint(b, shape, row, col)
                let (cleared, eroded, nLines) = clear(painted, shape, row)
                let landing = shape.map { height - (row + $0.r) }.max()!
                var rowT = 0
                for line in cleared {
                    var prev: UInt8 = 1
                    for cell in line { if cell != prev { rowT += 1 }; prev = cell }
                    if prev != 1 { rowT += 1 }
                }
                var colT = 0
                for c in 0..<width {
                    var prev: UInt8 = 0
                    for r in 0..<height { let cell = cleared[r][c]; if cell != prev { colT += 1 }; prev = cell }
                    if prev != 1 { colT += 1 }
                }
                var holes = 0
                for c in 0..<width {
                    var seen = false
                    for r in 0..<height { if cleared[r][c] != 0 { seen = true } else if seen { holes += 1 } }
                }
                var wells = 0
                for c in 0..<width {
                    let l = c == 0 ? height : columnHeight(cleared, c - 1)
                    let r = c == width - 1 ? height : columnHeight(cleared, c + 1)
                    let depth = min(l, r) - columnHeight(cleared, c)
                    if depth > 0 { wells += depth * (depth + 1) / 2 }
                }
                let rating = wLanding * Double(landing) + wEroded * Double(eroded) + wRowT * Double(rowT)
                    + wColT * Double(colT) + wHoles * Double(holes) + wWells * Double(wells)
                found.append(Placement(key: "r\(rotation)c\(col)", rotation: rotation, column: col, row: row, rating: rating,
                                       board: cleared, lines: nLines, holes: holes, shape: shape))
            }
        }
        return found
    }

    static func oracleKey(_ options: [Placement]) -> String? {
        options.max { a, b in
            (a.rating, -a.rotation, -a.column) < (b.rating, -b.rotation, -b.column)
        }?.key
    }

    private static let turns = ["unrotated", "rotated right", "rotated twice", "rotated left"]
    static func describe(_ piece: String, _ it: Placement) -> String {
        let cols = it.shape.map { it.column + $0.c }
        let lo = cols.min()!, hi = cols.max()!
        let span = lo == hi ? "column \(lo)" : "columns \(lo)-\(hi)"
        let bottom = it.row + it.shape.map(\.r).max()!
        return "\(piece) piece, \(turns[it.rotation]), \(span), lands on row \(bottom), "
            + "clears \(it.lines) \(it.lines == 1 ? "line" : "lines"), leaves \(it.holes) \(it.holes == 1 ? "hole" : "holes")"
    }
}
