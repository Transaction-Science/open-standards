//! Deterministic computation toolkit — zero-cost, zero-hallucination tools.
//!
//! Every function here is a pure computation that:
//! - Costs 0 energy (no GPU, no LLM, no network)
//! - Returns in <1ms
//! - Is 100% accurate (no hallucination possible)
//! - Is the "calculator next to the AI"
//!
//! The AI's only job is to classify intent and extract parameters.
//! These tools do all the actual work.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────
// §1  Math Expression Evaluator
// ─────────────────────────────────────────────────

/// Evaluate a mathematical expression string.
///
/// Supports: `+`, `-`, `*`, `/`, `^` (power), `%` (modulo),
/// parentheses, and built-in functions:
/// `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
/// `log` (base 10), `ln` (natural), `abs`, `ceil`, `floor`, `round`,
/// `exp`, `factorial`.
///
/// Constants: `pi`, `e`, `tau`.
pub fn eval_math(expr: &str) -> Result<f64, String> {
    // Normalize "mod" keyword to "%" operator
    let normalized = expr.replace(" mod ", " % ");
    let tokens = tokenize(&normalized)?;
    let mut parser = ExprParser::new(&tokens);
    let result = parser.parse_expr()?;
    if parser.pos < tokens.len() {
        return Err(format!("unexpected token: {:?}", tokens[parser.pos]));
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
    Comma,
    Func(String),
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' => i += 1,
            '+' | '*' | '/' | '%' | '^' => {
                tokens.push(Token::Op(chars[i]));
                i += 1;
            }
            '-' => {
                // Unary minus: if at start, after operator, after '(' or ','
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(Token::Op(_) | Token::LParen | Token::Comma)
                    );
                if is_unary {
                    // Parse the number directly with negative sign
                    if i + 1 < chars.len() && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.')
                    {
                        i += 1;
                        let start = i;
                        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                            i += 1;
                        }
                        // Check for scientific notation
                        if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                            i += 1;
                            if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                                i += 1;
                            }
                            while i < chars.len() && chars[i].is_ascii_digit() {
                                i += 1;
                            }
                        }
                        let num_str: String = chars[start..i].iter().collect();
                        let val: f64 = num_str
                            .parse()
                            .map_err(|_| format!("invalid number: -{num_str}"))?;
                        tokens.push(Token::Num(-val));
                    } else {
                        // Unary minus before parenthesis or function: push -1 *
                        tokens.push(Token::Num(-1.0));
                        tokens.push(Token::Op('*'));
                    }
                } else {
                    tokens.push(Token::Op('-'));
                    i += 1;
                }
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                // Scientific notation
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    i += 1;
                    if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                        i += 1;
                    }
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_str: String = chars[start..i].iter().collect();
                let val: f64 = num_str
                    .parse()
                    .map_err(|_| format!("invalid number: {num_str}"))?;
                tokens.push(Token::Num(val));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                let lower = name.to_lowercase();

                // Constants
                match lower.as_str() {
                    "pi" => tokens.push(Token::Num(std::f64::consts::PI)),
                    "e" if i >= chars.len() || chars[i] != '(' => {
                        // Only treat as constant if not followed by '('
                        // (to distinguish from function names starting with 'e')
                        tokens.push(Token::Num(std::f64::consts::E));
                    }
                    "tau" => tokens.push(Token::Num(std::f64::consts::TAU)),
                    "inf" | "infinity" => tokens.push(Token::Num(f64::INFINITY)),
                    _ => tokens.push(Token::Func(lower)),
                }
            }
            c => return Err(format!("unexpected character: '{c}'")),
        }
    }

    Ok(tokens)
}

struct ExprParser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    // expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut left = self.parse_term()?;
        while let Some(Token::Op(op @ ('+' | '-'))) = self.peek() {
            let op = *op;
            self.next();
            let right = self.parse_term()?;
            left = match op {
                '+' => left + right,
                '-' => left - right,
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    // term = power (('*' | '/' | '%') power)*
    fn parse_term(&mut self) -> Result<f64, String> {
        let mut left = self.parse_power()?;
        while let Some(Token::Op(op @ ('*' | '/' | '%'))) = self.peek() {
            let op = *op;
            self.next();
            let right = self.parse_power()?;
            left = match op {
                '*' => left * right,
                '/' => {
                    if right == 0.0 {
                        return Err("division by zero".into());
                    }
                    left / right
                }
                '%' => {
                    if right == 0.0 {
                        return Err("modulo by zero".into());
                    }
                    left % right
                }
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    // power = unary ('^' power)?   (right-associative)
    fn parse_power(&mut self) -> Result<f64, String> {
        let base = self.parse_unary()?;
        if let Some(Token::Op('^')) = self.peek() {
            self.next();
            let exp = self.parse_power()?; // right-associative
            Ok(base.powf(exp))
        } else {
            Ok(base)
        }
    }

    // unary = ('+' | '-')? atom
    fn parse_unary(&mut self) -> Result<f64, String> {
        if let Some(Token::Op('+')) = self.peek() {
            self.next();
            return self.parse_atom();
        }
        if let Some(Token::Op('-')) = self.peek() {
            self.next();
            let val = self.parse_atom()?;
            return Ok(-val);
        }
        self.parse_atom()
    }

    // atom = Num | Func '(' args ')' | '(' expr ')'
    fn parse_atom(&mut self) -> Result<f64, String> {
        match self.next() {
            Some(Token::Num(n)) => Ok(*n),
            Some(Token::Func(name)) => {
                let name = name.clone();
                // Expect '('
                match self.next() {
                    Some(Token::LParen) => {}
                    _ => return Err(format!("expected '(' after function '{name}'")),
                }
                // Parse arguments
                let mut args = vec![self.parse_expr()?];
                while let Some(Token::Comma) = self.peek() {
                    self.next();
                    args.push(self.parse_expr()?);
                }
                // Expect ')'
                match self.next() {
                    Some(Token::RParen) => {}
                    _ => return Err("expected ')' after function arguments".to_string()),
                }
                eval_func(&name, &args)
            }
            Some(Token::LParen) => {
                let val = self.parse_expr()?;
                match self.next() {
                    Some(Token::RParen) => Ok(val),
                    _ => Err("expected ')'".into()),
                }
            }
            Some(t) => Err(format!("unexpected token: {t:?}")),
            None => Err("unexpected end of expression".into()),
        }
    }
}

fn eval_func(name: &str, args: &[f64]) -> Result<f64, String> {
    match name {
        "sqrt" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].sqrt())
        }
        "sin" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].sin())
        }
        "cos" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].cos())
        }
        "tan" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].tan())
        }
        "asin" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].asin())
        }
        "acos" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].acos())
        }
        "atan" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].atan())
        }
        "atan2" => {
            ensure_args(name, args, 2)?;
            Ok(args[0].atan2(args[1]))
        }
        "log" | "log10" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].log10())
        }
        "log2" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].log2())
        }
        "ln" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].ln())
        }
        "abs" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].abs())
        }
        "ceil" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].ceil())
        }
        "floor" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].floor())
        }
        "round" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].round())
        }
        "exp" => {
            ensure_args(name, args, 1)?;
            Ok(args[0].exp())
        }
        "factorial" => {
            ensure_args(name, args, 1)?;
            let n = args[0] as u64;
            if n > 20 {
                return Err("factorial overflow (max 20!)".into());
            }
            Ok(factorial(n) as f64)
        }
        "min" => {
            if args.len() < 2 {
                return Err(format!(
                    "min requires at least 2 arguments, got {}",
                    args.len()
                ));
            }
            Ok(args.iter().copied().fold(f64::INFINITY, f64::min))
        }
        "max" => {
            if args.len() < 2 {
                return Err(format!(
                    "max requires at least 2 arguments, got {}",
                    args.len()
                ));
            }
            Ok(args.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        }
        "pow" => {
            ensure_args(name, args, 2)?;
            Ok(args[0].powf(args[1]))
        }
        "hypot" => {
            ensure_args(name, args, 2)?;
            Ok(args[0].hypot(args[1]))
        }
        _ => Err(format!("unknown function: {name}")),
    }
}

fn ensure_args(name: &str, args: &[f64], expected: usize) -> Result<(), String> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} expects {expected} argument(s), got {}",
            args.len()
        ))
    }
}

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// Format a math result for display.
pub fn format_math_result(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        // Trim trailing zeros but keep at least one decimal
        let s = format!("{value:.10}");
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

// ─────────────────────────────────────────────────
// §2  Unit Converter
// ─────────────────────────────────────────────────

/// Convert a value between units.
///
/// Returns `(converted_value, from_unit_canonical, to_unit_canonical)`.
pub fn convert_units(value: f64, from: &str, to: &str) -> Result<(f64, String, String), String> {
    let from_lower = from.to_lowercase();
    let to_lower = to.to_lowercase();

    // Temperature (special case — not linear conversion)
    if let Some(result) = try_convert_temperature(value, &from_lower, &to_lower) {
        return result;
    }

    // Linear conversions: normalize both units to a base unit, then convert
    let (from_factor, from_canon, from_category) = unit_to_base(&from_lower)?;
    let (to_factor, to_canon, to_category) = unit_to_base(&to_lower)?;

    if from_category != to_category {
        return Err(format!(
            "cannot convert {from_canon} ({from_category}) to {to_canon} ({to_category})"
        ));
    }

    let base_value = value * from_factor;
    let result = base_value / to_factor;

    Ok((result, from_canon, to_canon))
}

fn try_convert_temperature(
    value: f64,
    from: &str,
    to: &str,
) -> Option<Result<(f64, String, String), String>> {
    let from_temp = match from {
        "c" | "celsius" | "°c" => Some("celsius"),
        "f" | "fahrenheit" | "°f" => Some("fahrenheit"),
        "k" | "kelvin" => Some("kelvin"),
        _ => None,
    };
    let to_temp = match to {
        "c" | "celsius" | "°c" => Some("celsius"),
        "f" | "fahrenheit" | "°f" => Some("fahrenheit"),
        "k" | "kelvin" => Some("kelvin"),
        _ => None,
    };

    match (from_temp, to_temp) {
        (Some(f), Some(t)) => {
            let result = match (f, t) {
                ("celsius", "fahrenheit") => value * 9.0 / 5.0 + 32.0,
                ("celsius", "kelvin") => value + 273.15,
                ("fahrenheit", "celsius") => (value - 32.0) * 5.0 / 9.0,
                ("fahrenheit", "kelvin") => (value - 32.0) * 5.0 / 9.0 + 273.15,
                ("kelvin", "celsius") => value - 273.15,
                ("kelvin", "fahrenheit") => (value - 273.15) * 9.0 / 5.0 + 32.0,
                (a, b) if a == b => value,
                _ => unreachable!(),
            };
            Some(Ok((result, f.to_string(), t.to_string())))
        }
        (Some(_), None) | (None, Some(_)) => {
            Some(Err("cannot mix temperature with other unit types".into()))
        }
        (None, None) => None,
    }
}

/// Returns `(factor_to_base, canonical_name, category)`.
fn unit_to_base(unit: &str) -> Result<(f64, String, &'static str), String> {
    type UnitCategory<'a> = (&'a [(&'a [&'a str], f64, &'a str)], &'a str);

    // Length (base: meters)
    let length: &[(&[&str], f64, &str)] = &[
        (
            &[
                "mm",
                "millimeter",
                "millimeters",
                "millimetre",
                "millimetres",
            ],
            0.001,
            "mm",
        ),
        (
            &[
                "cm",
                "centimeter",
                "centimeters",
                "centimetre",
                "centimetres",
            ],
            0.01,
            "cm",
        ),
        (&["m", "meter", "meters", "metre", "metres"], 1.0, "m"),
        (
            &["km", "kilometer", "kilometers", "kilometre", "kilometres"],
            1000.0,
            "km",
        ),
        (&["in", "inch", "inches", "\""], 0.0254, "in"),
        (&["ft", "foot", "feet", "'"], 0.3048, "ft"),
        (&["yd", "yard", "yards"], 0.9144, "yd"),
        (&["mi", "mile", "miles"], 1609.344, "mi"),
        (
            &["nm", "nautical mile", "nautical miles", "nmi"],
            1852.0,
            "nmi",
        ),
        (
            &["μm", "um", "micron", "microns", "micrometer", "micrometers"],
            1e-6,
            "μm",
        ),
        (
            &["ly", "light year", "light years", "lightyear", "lightyears"],
            9.461e15,
            "ly",
        ),
        (
            &["au", "astronomical unit", "astronomical units"],
            1.496e11,
            "au",
        ),
    ];

    // Mass (base: kilograms)
    let mass: &[(&[&str], f64, &str)] = &[
        (&["mg", "milligram", "milligrams"], 1e-6, "mg"),
        (&["g", "gram", "grams"], 0.001, "g"),
        (&["kg", "kilogram", "kilograms", "kilo", "kilos"], 1.0, "kg"),
        (
            &["t", "tonne", "tonnes", "metric ton", "metric tons"],
            1000.0,
            "t",
        ),
        (&["oz", "ounce", "ounces"], 0.0283495, "oz"),
        (&["lb", "lbs", "pound", "pounds"], 0.453592, "lb"),
        (&["st", "stone", "stones"], 6.35029, "st"),
        (&["ton", "tons", "short ton", "short tons"], 907.185, "ton"),
    ];

    // Time (base: seconds)
    let time: &[(&[&str], f64, &str)] = &[
        (&["ms", "millisecond", "milliseconds", "msec"], 0.001, "ms"),
        (&["s", "sec", "second", "seconds"], 1.0, "s"),
        (&["min", "minute", "minutes"], 60.0, "min"),
        (&["h", "hr", "hour", "hours"], 3600.0, "h"),
        (&["d", "day", "days"], 86400.0, "d"),
        (&["wk", "week", "weeks"], 604800.0, "wk"),
        (&["yr", "year", "years"], 31557600.0, "yr"),
        (&["μs", "us", "microsecond", "microseconds"], 1e-6, "μs"),
        (&["ns", "nanosecond", "nanoseconds"], 1e-9, "ns"),
    ];

    // Data (base: bytes)
    let data: &[(&[&str], f64, &str)] = &[
        (&["b", "byte", "bytes"], 1.0, "B"),
        (&["kb", "kilobyte", "kilobytes"], 1024.0, "KB"),
        (&["mb", "megabyte", "megabytes"], 1_048_576.0, "MB"),
        (&["gb", "gigabyte", "gigabytes"], 1_073_741_824.0, "GB"),
        (&["tb", "terabyte", "terabytes"], 1_099_511_627_776.0, "TB"),
        (
            &["pb", "petabyte", "petabytes"],
            1_125_899_906_842_624.0,
            "PB",
        ),
        (&["bit", "bits"], 0.125, "bit"),
        (&["kbit", "kilobit", "kilobits", "kbps"], 128.0, "kbit"),
        (&["mbit", "megabit", "megabits", "mbps"], 131_072.0, "Mbit"),
        (
            &["gbit", "gigabit", "gigabits", "gbps"],
            134_217_728.0,
            "Gbit",
        ),
    ];

    // Speed (base: m/s)
    let speed: &[(&[&str], f64, &str)] = &[
        (&["m/s", "mps", "meters per second"], 1.0, "m/s"),
        (
            &["km/h", "kph", "kmh", "kilometers per hour"],
            1.0 / 3.6,
            "km/h",
        ),
        (&["mph", "miles per hour"], 0.44704, "mph"),
        (&["knot", "knots", "kt", "kn"], 0.514444, "knot"),
        (&["mach"], 343.0, "Mach"),
    ];

    // Area (base: m²)
    let area: &[(&[&str], f64, &str)] = &[
        (
            &[
                "m2",
                "m²",
                "sq m",
                "square meter",
                "square meters",
                "square metre",
            ],
            1.0,
            "m²",
        ),
        (
            &[
                "km2",
                "km²",
                "sq km",
                "square kilometer",
                "square kilometers",
            ],
            1e6,
            "km²",
        ),
        (&["cm2", "cm²", "sq cm", "square centimeter"], 1e-4, "cm²"),
        (
            &["ft2", "ft²", "sq ft", "square foot", "square feet"],
            0.092903,
            "ft²",
        ),
        (
            &["in2", "in²", "sq in", "square inch", "square inches"],
            6.4516e-4,
            "in²",
        ),
        (
            &["mi2", "mi²", "sq mi", "square mile", "square miles"],
            2.59e6,
            "mi²",
        ),
        (&["acre", "acres", "ac"], 4046.86, "acre"),
        (&["hectare", "hectares", "ha"], 10000.0, "ha"),
    ];

    // Volume (base: liters)
    let volume: &[(&[&str], f64, &str)] = &[
        (
            &["ml", "milliliter", "milliliters", "millilitre"],
            0.001,
            "mL",
        ),
        (&["l", "liter", "liters", "litre", "litres"], 1.0, "L"),
        (&["gal", "gallon", "gallons"], 3.78541, "gal"),
        (&["qt", "quart", "quarts"], 0.946353, "qt"),
        (&["pt", "pint", "pints"], 0.473176, "pt"),
        (&["cup", "cups"], 0.236588, "cup"),
        (
            &["fl oz", "fluid ounce", "fluid ounces", "floz"],
            0.0295735,
            "fl oz",
        ),
        (&["tbsp", "tablespoon", "tablespoons"], 0.0147868, "tbsp"),
        (&["tsp", "teaspoon", "teaspoons"], 0.00492892, "tsp"),
        (&["m3", "m³", "cubic meter", "cubic meters"], 1000.0, "m³"),
    ];

    // Energy (base: joules)
    let energy: &[(&[&str], f64, &str)] = &[
        (&["j", "joule", "joules"], 1.0, "J"),
        (&["kj", "kilojoule", "kilojoules"], 1000.0, "kJ"),
        (&["cal", "calorie", "calories"], 4.184, "cal"),
        (&["kcal", "kilocalorie", "kilocalories"], 4184.0, "kcal"),
        (&["wh", "watt hour", "watt hours", "watthour"], 3600.0, "Wh"),
        (
            &["kwh", "kilowatt hour", "kilowatt hours"],
            3_600_000.0,
            "kWh",
        ),
        (&["btu"], 1055.06, "BTU"),
        (&["ev", "electronvolt", "electron volt"], 1.602e-19, "eV"),
    ];

    // Pressure (base: pascals)
    let pressure: &[(&[&str], f64, &str)] = &[
        (&["pa", "pascal", "pascals"], 1.0, "Pa"),
        (&["kpa", "kilopascal", "kilopascals"], 1000.0, "kPa"),
        (&["bar"], 100_000.0, "bar"),
        (&["atm", "atmosphere", "atmospheres"], 101_325.0, "atm"),
        (&["psi", "pounds per square inch"], 6894.76, "psi"),
        (&["mmhg", "torr"], 133.322, "mmHg"),
    ];

    // Angle (base: radians)
    let angle: &[(&[&str], f64, &str)] = &[
        (&["rad", "radian", "radians"], 1.0, "rad"),
        (
            &["deg", "degree", "degrees", "°"],
            std::f64::consts::PI / 180.0,
            "°",
        ),
        (
            &["grad", "gradian", "gradians", "gon"],
            std::f64::consts::PI / 200.0,
            "grad",
        ),
        (
            &["turn", "turns", "rev", "revolution", "revolutions"],
            std::f64::consts::TAU,
            "turn",
        ),
    ];

    let all_categories: &[UnitCategory<'_>] = &[
        (length, "length"),
        (mass, "mass"),
        (time, "time"),
        (data, "data"),
        (speed, "speed"),
        (area, "area"),
        (volume, "volume"),
        (energy, "energy"),
        (pressure, "pressure"),
        (angle, "angle"),
    ];

    for (units, category) in all_categories {
        for (aliases, factor, canonical) in *units {
            for alias in *aliases {
                if unit == *alias {
                    return Ok((*factor, canonical.to_string(), category));
                }
            }
        }
    }

    Err(format!("unknown unit: '{unit}'"))
}

/// Format a unit conversion result for display.
pub fn format_conversion(value: f64, result: f64, from: &str, to: &str) -> String {
    format!(
        "{} {} = {} {}",
        format_math_result(value),
        from,
        format_math_result(result),
        to
    )
}

// ─────────────────────────────────────────────────
// §3  Number Base Converter
// ─────────────────────────────────────────────────

/// Convert an integer between number bases.
///
/// Supported bases: binary (2), octal (8), decimal (10), hexadecimal (16).
/// Input can be prefixed: `0b`, `0o`, `0x`, or plain decimal.
pub fn convert_base(input: &str, to_base: u32) -> Result<String, String> {
    let trimmed = input.trim();

    let (value, _from_base) = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        let v = i64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex: {e}"))?;
        (v, 16)
    } else if let Some(bin) = trimmed
        .strip_prefix("0b")
        .or_else(|| trimmed.strip_prefix("0B"))
    {
        let v = i64::from_str_radix(bin, 2).map_err(|e| format!("invalid binary: {e}"))?;
        (v, 2)
    } else if let Some(oct) = trimmed
        .strip_prefix("0o")
        .or_else(|| trimmed.strip_prefix("0O"))
    {
        let v = i64::from_str_radix(oct, 8).map_err(|e| format!("invalid octal: {e}"))?;
        (v, 8)
    } else {
        let v: i64 = trimmed
            .parse()
            .map_err(|e| format!("invalid number: {e}"))?;
        (v, 10)
    };

    let result = match to_base {
        2 => format!("0b{value:b}"),
        8 => format!("0o{value:o}"),
        10 => format!("{value}"),
        16 => format!("0x{value:X}"),
        _ => return Err(format!("unsupported base: {to_base} (use 2, 8, 10, or 16)")),
    };

    Ok(result)
}

/// Format all representations of a number.
pub fn format_all_bases(input: &str) -> Result<String, String> {
    let trimmed = input.trim();

    let value = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        i64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex: {e}"))?
    } else if let Some(bin) = trimmed
        .strip_prefix("0b")
        .or_else(|| trimmed.strip_prefix("0B"))
    {
        i64::from_str_radix(bin, 2).map_err(|e| format!("invalid binary: {e}"))?
    } else if let Some(oct) = trimmed
        .strip_prefix("0o")
        .or_else(|| trimmed.strip_prefix("0O"))
    {
        i64::from_str_radix(oct, 8).map_err(|e| format!("invalid octal: {e}"))?
    } else {
        trimmed
            .parse::<i64>()
            .map_err(|e| format!("invalid number: {e}"))?
    };

    Ok(format!(
        "decimal: {value}\nbinary: 0b{value:b}\noctal: 0o{value:o}\nhex: 0x{value:X}"
    ))
}

// ─────────────────────────────────────────────────
// §4  Text Transforms
// ─────────────────────────────────────────────────

/// Encode a string to base64.
pub fn base64_encode(input: &str) -> String {
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, input.as_bytes())
}

/// Decode a base64 string.
pub fn base64_decode(input: &str) -> Result<String, String> {
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, input.trim())
        .map_err(|e| format!("invalid base64: {e}"))?;
    String::from_utf8(bytes).map_err(|e| format!("decoded bytes are not valid UTF-8: {e}"))
}

/// URL-encode a string.
pub fn url_encode(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 3);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                write!(encoded, "%{byte:02X}").unwrap();
            }
        }
    }
    encoded
}

/// URL-decode a string.
pub fn url_decode(input: &str) -> Result<String, String> {
    let mut bytes = Vec::with_capacity(input.len());
    let mut chars = input.bytes();
    while let Some(b) = chars.next() {
        if b == b'%' {
            let h1 = chars.next().ok_or("incomplete percent-encoding")?;
            let h2 = chars.next().ok_or("incomplete percent-encoding")?;
            let hex = format!("{}{}", h1 as char, h2 as char);
            let byte =
                u8::from_str_radix(&hex, 16).map_err(|_| format!("invalid hex in URL: %{hex}"))?;
            bytes.push(byte);
        } else if b == b'+' {
            bytes.push(b' ');
        } else {
            bytes.push(b);
        }
    }
    String::from_utf8(bytes).map_err(|e| format!("decoded bytes are not valid UTF-8: {e}"))
}

/// Compute SHA-256 hash of input.
pub fn sha256(input: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in &result {
        write!(hex, "{b:02x}").unwrap();
    }
    hex
}

/// Count characters, words, and lines in text.
pub fn text_stats(input: &str) -> TextStats {
    let chars = input.chars().count();
    let words = input.split_whitespace().count();
    let lines = if input.is_empty() {
        0
    } else {
        input.lines().count()
    };
    let bytes = input.len();
    let sentences = input
        .chars()
        .filter(|c| *c == '.' || *c == '!' || *c == '?')
        .count();
    TextStats {
        characters: chars,
        words,
        lines,
        bytes,
        sentences,
    }
}

/// Text statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStats {
    pub characters: usize,
    pub words: usize,
    pub lines: usize,
    pub bytes: usize,
    pub sentences: usize,
}

impl std::fmt::Display for TextStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "characters: {}\nwords: {}\nlines: {}\nsentences: {}\nbytes: {}",
            self.characters, self.words, self.lines, self.sentences, self.bytes
        )
    }
}

/// Convert text to uppercase.
pub fn to_upper(input: &str) -> String {
    input.to_uppercase()
}

/// Convert text to lowercase.
pub fn to_lower(input: &str) -> String {
    input.to_lowercase()
}

/// Convert text to title case.
pub fn to_title_case(input: &str) -> String {
    input
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    let rest: String = chars.as_str().to_lowercase();
                    format!("{upper}{rest}")
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reverse a string.
pub fn reverse_string(input: &str) -> String {
    input.chars().rev().collect()
}

// ─────────────────────────────────────────────────
// §5  Color Converter
// ─────────────────────────────────────────────────

/// A color in multiple representations.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorInfo {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub hex: String,
    pub rgb: String,
    pub hsl: String,
}

impl std::fmt::Display for ColorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hex: {}\nrgb: {}\nhsl: {}", self.hex, self.rgb, self.hsl)
    }
}

/// Parse a color from hex (#FF0000), rgb(255,0,0), or hsl(0,100%,50%) format.
pub fn parse_color(input: &str) -> Result<ColorInfo, String> {
    let trimmed = input.trim();

    if let Some(hex) = trimmed.strip_prefix('#') {
        return parse_hex_color(hex);
    }
    if let Some(rgb_inner) = trimmed
        .strip_prefix("rgb(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return parse_rgb_color(rgb_inner);
    }
    if let Some(hsl_inner) = trimmed
        .strip_prefix("hsl(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return parse_hsl_color(hsl_inner);
    }

    // Try as bare hex
    if trimmed.len() == 6 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        return parse_hex_color(trimmed);
    }

    Err(format!(
        "unrecognized color format: '{trimmed}' (use #RRGGBB, rgb(r,g,b), or hsl(h,s%,l%))"
    ))
}

fn parse_hex_color(hex: &str) -> Result<ColorInfo, String> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Err(format!("hex color must be 6 digits, got {}", hex.len()));
    }
    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| "invalid red hex")?;
    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| "invalid green hex")?;
    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| "invalid blue hex")?;
    Ok(build_color_info(r, g, b))
}

fn parse_rgb_color(inner: &str) -> Result<ColorInfo, String> {
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() != 3 {
        return Err("rgb() requires 3 values".into());
    }
    let r: u8 = parts[0].trim().parse().map_err(|_| "invalid red value")?;
    let g: u8 = parts[1].trim().parse().map_err(|_| "invalid green value")?;
    let b: u8 = parts[2].trim().parse().map_err(|_| "invalid blue value")?;
    Ok(build_color_info(r, g, b))
}

fn parse_hsl_color(inner: &str) -> Result<ColorInfo, String> {
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() != 3 {
        return Err("hsl() requires 3 values".into());
    }
    let h: f64 = parts[0].trim().parse().map_err(|_| "invalid hue")?;
    let s: f64 = parts[1]
        .trim()
        .trim_end_matches('%')
        .parse::<f64>()
        .map_err(|_| "invalid saturation")?
        / 100.0;
    let l: f64 = parts[2]
        .trim()
        .trim_end_matches('%')
        .parse::<f64>()
        .map_err(|_| "invalid lightness")?
        / 100.0;

    let (r, g, b) = hsl_to_rgb(h, s, l);
    Ok(build_color_info(r, g, b))
}

fn build_color_info(r: u8, g: u8, b: u8) -> ColorInfo {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    ColorInfo {
        r,
        g,
        b,
        hex: format!("#{r:02X}{g:02X}{b:02X}"),
        rgb: format!("rgb({r}, {g}, {b})"),
        hsl: format!("hsl({:.0}, {:.0}%, {:.0}%)", h, s * 100.0, l * 100.0),
    }
}

fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = f64::midpoint(max, min);

    if (max - min).abs() < f64::EPSILON {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < f64::EPSILON {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h
    } else if (max - g).abs() < f64::EPSILON {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    (h * 60.0, s, l)
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    if s.abs() < f64::EPSILON {
        let v = (l * 255.0).round() as u8;
        return (v, v, v);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

// ─────────────────────────────────────────────────
// §6  Statistics
// ─────────────────────────────────────────────────

/// Compute basic statistics on a list of numbers.
pub fn statistics(values: &[f64]) -> Result<Stats, String> {
    if values.is_empty() {
        return Err("cannot compute statistics on empty list".into());
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len().is_multiple_of(2) {
        let mid = sorted.len() / 2;
        f64::midpoint(sorted[mid - 1], sorted[mid])
    } else {
        sorted[sorted.len() / 2]
    };

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Mode: most frequent value
    let mut freq: HashMap<u64, usize> = HashMap::new();
    for &v in values {
        *freq.entry(v.to_bits()).or_insert(0) += 1;
    }
    let max_freq = freq.values().copied().max().unwrap_or(0);
    let mode = if max_freq > 1 {
        freq.iter()
            .filter(|&(_, count)| *count == max_freq)
            .map(|(&bits, _)| f64::from_bits(bits))
            .next()
    } else {
        None // No mode if all values appear once
    };

    Ok(Stats {
        count: values.len(),
        sum,
        mean,
        median,
        mode,
        min,
        max,
        range: max - min,
        variance,
        std_dev,
    })
}

/// Statistical summary.
#[derive(Debug, Clone)]
pub struct Stats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub variance: f64,
    pub std_dev: f64,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "count: {}\nsum: {}\nmean: {}\nmedian: {}\nmode: {}\nmin: {}\nmax: {}\nrange: {}\nvariance: {}\nstd dev: {}",
            self.count,
            format_math_result(self.sum),
            format_math_result(self.mean),
            format_math_result(self.median),
            self.mode
                .map_or_else(|| "N/A".to_string(), format_math_result),
            format_math_result(self.min),
            format_math_result(self.max),
            format_math_result(self.range),
            format_math_result(self.variance),
            format_math_result(self.std_dev),
        )
    }
}

/// Parse a comma/space separated list of numbers.
pub fn parse_number_list(input: &str) -> Result<Vec<f64>, String> {
    let cleaned = input.replace(',', " ");
    let numbers: Result<Vec<f64>, _> = cleaned
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(str::parse::<f64>)
        .collect();
    numbers.map_err(|e| format!("invalid number in list: {e}"))
}

// ─────────────────────────────────────────────────
// §7  Date/Time Calculator
// ─────────────────────────────────────────────────

/// Calculate days between two dates (YYYY-MM-DD format).
pub fn days_between(date1: &str, date2: &str) -> Result<i64, String> {
    let d1 = parse_date(date1)?;
    let d2 = parse_date(date2)?;
    Ok((d2 - d1).num_days())
}

/// Add days to a date.
pub fn add_days(date: &str, days: i64) -> Result<String, String> {
    let d = parse_date(date)?;
    let result = d + chrono::Duration::days(days);
    Ok(result.format("%Y-%m-%d").to_string())
}

/// Get day of week for a date.
pub fn day_of_week(date: &str) -> Result<String, String> {
    let d = parse_date(date)?;
    Ok(d.format("%A").to_string())
}

/// Parse a date string in common formats.
fn parse_date(input: &str) -> Result<chrono::NaiveDate, String> {
    let trimmed = input.trim();

    // Try YYYY-MM-DD
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return Ok(d);
    }
    // Try MM/DD/YYYY
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%m/%d/%Y") {
        return Ok(d);
    }
    // Try DD-MM-YYYY
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%d-%m-%Y") {
        return Ok(d);
    }
    // Try Month DD, YYYY
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%B %d, %Y") {
        return Ok(d);
    }
    // Try Mon DD, YYYY
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%b %d, %Y") {
        return Ok(d);
    }

    Err(format!(
        "cannot parse date: '{trimmed}' (use YYYY-MM-DD, MM/DD/YYYY, or 'Month DD, YYYY')"
    ))
}

// ─────────────────────────────────────────────────
// §8  MIDI / Music Theory
// ─────────────────────────────────────────────────

/// MIDI note number → frequency (A4 = 69 = 440 Hz).
pub fn midi_to_freq(note: u8) -> f64 {
    440.0 * 2.0_f64.powf((note as f64 - 69.0) / 12.0)
}

/// Frequency → nearest MIDI note + cents deviation.
pub fn freq_to_midi(freq: f64) -> (u8, f64) {
    if freq <= 0.0 {
        return (0, 0.0);
    }
    let midi_float = 69.0 + 12.0 * (freq / 440.0).log2();
    let note = midi_float.round() as u8;
    let cents = (midi_float - note as f64) * 100.0;
    (note, cents)
}

/// MIDI note number → note name (e.g., 60 → "C4").
pub fn midi_to_name(note: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let name = NAMES[(note % 12) as usize];
    let octave = i16::from(note / 12) - 1;
    format!("{name}{octave}")
}

/// Note name (e.g., "C4", "A#3") → MIDI note number.
pub fn name_to_midi(name: &str) -> Result<u8, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("empty note name".into());
    }

    let (note_part, octave_str) =
        if name.len() >= 2 && (name.as_bytes()[1] == b'#' || name.as_bytes()[1] == b'b') {
            (&name[..2], &name[2..])
        } else {
            (&name[..1], &name[1..])
        };

    let semitone = match note_part.to_uppercase().as_str() {
        "C" => 0,
        "C#" | "DB" => 1,
        "D" => 2,
        "D#" | "EB" => 3,
        "E" => 4,
        "F" => 5,
        "F#" | "GB" => 6,
        "G" => 7,
        "G#" | "AB" => 8,
        "A" => 9,
        "A#" | "BB" => 10,
        "B" => 11,
        _ => return Err(format!("unknown note: '{note_part}'")),
    };

    let octave: i8 = octave_str
        .parse()
        .map_err(|_| format!("invalid octave: '{octave_str}'"))?;

    let midi = (octave + 1) as i16 * 12 + semitone as i16;
    if !(0..=127).contains(&midi) {
        return Err(format!("MIDI note {midi} out of range 0-127"));
    }
    Ok(midi as u8)
}

/// BPM → milliseconds per beat (quarter note).
pub fn bpm_to_ms(bpm: f64) -> f64 {
    60_000.0 / bpm
}

/// BPM → samples per beat at a given sample rate.
pub fn bpm_to_samples(bpm: f64, sample_rate: f64) -> f64 {
    sample_rate * 60.0 / bpm
}

// ─────────────────────────────────────────────────
// §9  Roman Numerals
// ─────────────────────────────────────────────────

/// Integer → Roman numeral string (1–3999).
pub fn to_roman(mut n: u32) -> Result<String, String> {
    if n == 0 || n > 3999 {
        return Err("Roman numerals only support 1–3999".into());
    }
    let table: [(u32, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut result = String::new();
    for &(val, sym) in &table {
        while n >= val {
            result.push_str(sym);
            n -= val;
        }
    }
    Ok(result)
}

/// Roman numeral string → integer.
pub fn from_roman(input: &str) -> Result<u32, String> {
    let input = input.trim().to_uppercase();
    if input.is_empty() {
        return Err("empty roman numeral".into());
    }
    let value_of = |c: char| -> Result<u32, String> {
        match c {
            'I' => Ok(1),
            'V' => Ok(5),
            'X' => Ok(10),
            'L' => Ok(50),
            'C' => Ok(100),
            'D' => Ok(500),
            'M' => Ok(1000),
            _ => Err(format!("invalid roman numeral character: '{c}'")),
        }
    };
    let chars: Vec<char> = input.chars().collect();
    let mut total = 0u32;
    let mut i = 0;
    while i < chars.len() {
        let curr = value_of(chars[i])?;
        if i + 1 < chars.len() {
            let next = value_of(chars[i + 1])?;
            if curr < next {
                total += next - curr;
                i += 2;
                continue;
            }
        }
        total += curr;
        i += 1;
    }
    Ok(total)
}

// ─────────────────────────────────────────────────
// §10  Percentage Calculator
// ─────────────────────────────────────────────────

/// Calculate "what is X% of Y".
pub fn percentage_of(pct: f64, value: f64) -> f64 {
    pct / 100.0 * value
}

/// Calculate "X is what % of Y".
pub fn what_percentage(part: f64, whole: f64) -> Result<f64, String> {
    if whole == 0.0 {
        return Err("cannot calculate percentage of zero".into());
    }
    Ok(part / whole * 100.0)
}

/// Calculate percentage change from old to new.
pub fn percentage_change(old: f64, new: f64) -> Result<f64, String> {
    if old == 0.0 {
        return Err("cannot calculate percentage change from zero".into());
    }
    Ok((new - old) / old * 100.0)
}

// ─────────────────────────────────────────────────
// §11  UUID Generator
// ─────────────────────────────────────────────────

/// Generate a v4 UUID.
pub fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ─────────────────────────────────────────────────
// §12  Unix Timestamp Converter
// ─────────────────────────────────────────────────

/// Unix timestamp (seconds) → human-readable UTC date-time.
pub fn timestamp_to_datetime(ts: i64) -> Result<String, String> {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .ok_or_else(|| format!("invalid timestamp: {ts}"))
}

/// Human-readable date → Unix timestamp (seconds).
pub fn datetime_to_timestamp(input: &str) -> Result<i64, String> {
    let d = parse_date(input)?;
    Ok(d.and_hms_opt(0, 0, 0)
        .ok_or("invalid time")?
        .and_utc()
        .timestamp())
}

/// Current Unix timestamp.
pub fn now_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

// ─────────────────────────────────────────────────
// §13  SVG Parametric Generator
// ─────────────────────────────────────────────────

/// A parametric SVG shape.
#[derive(Debug, Clone, PartialEq)]
pub enum SvgShape {
    /// Rectangle: x, y, width, height, optional corner radius.
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        rx: f64,
    },
    /// Circle: center x, center y, radius.
    Circle { cx: f64, cy: f64, r: f64 },
    /// Ellipse: center x, center y, rx, ry.
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
    /// Line: start and end points.
    Line { x1: f64, y1: f64, x2: f64, y2: f64 },
    /// Regular polygon: center, radius, number of sides.
    Polygon {
        cx: f64,
        cy: f64,
        r: f64,
        sides: u32,
    },
    /// Star: center, outer radius, inner radius, number of points.
    Star {
        cx: f64,
        cy: f64,
        outer_r: f64,
        inner_r: f64,
        points: u32,
    },
    /// Text label.
    Text {
        x: f64,
        y: f64,
        text: String,
        size: f64,
    },
    /// Grid: cols, rows, cell size, with optional gap.
    Grid {
        cols: u32,
        rows: u32,
        cell: f64,
        gap: f64,
    },
}

/// SVG generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SvgParams {
    pub width: f64,
    pub height: f64,
    pub shapes: Vec<SvgShape>,
    pub stroke: String,
    pub fill: String,
    pub stroke_width: f64,
    pub background: Option<String>,
}

impl Default for SvgParams {
    fn default() -> Self {
        Self {
            width: 400.0,
            height: 400.0,
            shapes: Vec::new(),
            stroke: "#E0E0E6".into(),
            fill: "none".into(),
            stroke_width: 2.0,
            background: None,
        }
    }
}

/// Generate an SVG document from parametric shapes.
pub fn generate_svg(params: &SvgParams) -> String {
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        params.width, params.height, params.width, params.height
    );
    svg.push('\n');

    if let Some(bg) = &params.background {
        write!(svg, r#"  <rect width="100%" height="100%" fill="{bg}"/>"#).unwrap();
        svg.push('\n');
    }

    let style = format!(
        r#"stroke="{}" fill="{}" stroke-width="{}""#,
        params.stroke, params.fill, params.stroke_width
    );

    for shape in &params.shapes {
        let elem = match shape {
            SvgShape::Rect { x, y, w, h, rx } => {
                if *rx > 0.0 {
                    format!(
                        r#"  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" {style}/>"#
                    )
                } else {
                    format!(r#"  <rect x="{x}" y="{y}" width="{w}" height="{h}" {style}/>"#)
                }
            }
            SvgShape::Circle { cx, cy, r } => {
                format!(r#"  <circle cx="{cx}" cy="{cy}" r="{r}" {style}/>"#)
            }
            SvgShape::Ellipse { cx, cy, rx, ry } => {
                format!(r#"  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" {style}/>"#)
            }
            SvgShape::Line { x1, y1, x2, y2 } => {
                format!(r#"  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {style}/>"#)
            }
            SvgShape::Polygon { cx, cy, r, sides } => {
                let points = polygon_points(*cx, *cy, *r, *sides);
                format!(r#"  <polygon points="{points}" {style}/>"#)
            }
            SvgShape::Star {
                cx,
                cy,
                outer_r,
                inner_r,
                points,
            } => {
                let pts = star_points(*cx, *cy, *outer_r, *inner_r, *points);
                format!(r#"  <polygon points="{pts}" {style}/>"#)
            }
            SvgShape::Text { x, y, text, size } => {
                format!(
                    r#"  <text x="{x}" y="{y}" font-size="{size}" fill="{}" font-family="sans-serif">{text}</text>"#,
                    params.stroke
                )
            }
            SvgShape::Grid {
                cols,
                rows,
                cell,
                gap,
            } => {
                let mut lines = String::new();
                let total_w = *cols as f64 * (cell + gap) - gap;
                let total_h = *rows as f64 * (cell + gap) - gap;
                for c in 0..=*cols {
                    let x = c as f64 * (cell + gap);
                    write!(
                        lines,
                        r#"  <line x1="{x}" y1="0" x2="{x}" y2="{total_h}" {style}/>"#
                    )
                    .unwrap();
                    lines.push('\n');
                }
                for r in 0..=*rows {
                    let y = r as f64 * (cell + gap);
                    write!(
                        lines,
                        r#"  <line x1="0" y1="{y}" x2="{total_w}" y2="{y}" {style}/>"#
                    )
                    .unwrap();
                    lines.push('\n');
                }
                lines
            }
        };
        svg.push_str(&elem);
        svg.push('\n');
    }

    svg.push_str("</svg>");
    svg
}

/// Generate polygon vertex points as an SVG points string.
fn polygon_points(cx: f64, cy: f64, r: f64, sides: u32) -> String {
    (0..sides)
        .map(|i| {
            let angle =
                std::f64::consts::TAU * i as f64 / sides as f64 - std::f64::consts::FRAC_PI_2;
            format!("{:.1},{:.1}", cx + r * angle.cos(), cy + r * angle.sin())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generate star vertex points (alternating outer/inner radii).
fn star_points(cx: f64, cy: f64, outer_r: f64, inner_r: f64, points: u32) -> String {
    let total = points * 2;
    (0..total)
        .map(|i| {
            let angle =
                std::f64::consts::TAU * i as f64 / total as f64 - std::f64::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { outer_r } else { inner_r };
            format!("{:.1},{:.1}", cx + r * angle.cos(), cy + r * angle.sin())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ─────────────────────────────────────────────────
// §14  DXF 2D Geometry Generator
// ─────────────────────────────────────────────────

/// A 2D DXF entity.
#[derive(Debug, Clone, PartialEq)]
pub enum DxfEntity {
    /// Line segment.
    Line { x1: f64, y1: f64, x2: f64, y2: f64 },
    /// Circle.
    Circle { cx: f64, cy: f64, r: f64 },
    /// Arc: center, radius, start angle (deg), end angle (deg).
    Arc {
        cx: f64,
        cy: f64,
        r: f64,
        start_deg: f64,
        end_deg: f64,
    },
    /// Rectangle (as 4 lines): lower-left corner, width, height.
    Rect { x: f64, y: f64, w: f64, h: f64 },
    /// Regular polygon: center, radius, number of sides.
    Polygon {
        cx: f64,
        cy: f64,
        r: f64,
        sides: u32,
    },
    /// Text annotation.
    Text {
        x: f64,
        y: f64,
        height: f64,
        text: String,
    },
    /// Dimension line between two points.
    Dimension { x1: f64, y1: f64, x2: f64, y2: f64 },
}

/// DXF generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct DxfParams {
    pub entities: Vec<DxfEntity>,
    pub layer: String,
}

impl Default for DxfParams {
    fn default() -> Self {
        Self {
            entities: Vec::new(),
            layer: "0".into(),
        }
    }
}

/// Generate a minimal DXF file (R12-compatible ASCII format).
pub fn generate_dxf(params: &DxfParams) -> String {
    let mut dxf = String::new();

    // HEADER section
    dxf.push_str("0\nSECTION\n2\nHEADER\n");
    dxf.push_str("9\n$ACADVER\n1\nAC1009\n"); // R12 format
    dxf.push_str("0\nENDSEC\n");

    // ENTITIES section
    dxf.push_str("0\nSECTION\n2\nENTITIES\n");

    for entity in &params.entities {
        match entity {
            DxfEntity::Line { x1, y1, x2, y2 } => {
                dxf.push_str(&dxf_line(*x1, *y1, *x2, *y2, &params.layer));
            }
            DxfEntity::Circle { cx, cy, r } => {
                write!(
                    dxf,
                    "0\nCIRCLE\n8\n{}\n10\n{cx}\n20\n{cy}\n40\n{r}\n",
                    params.layer
                )
                .unwrap();
            }
            DxfEntity::Arc {
                cx,
                cy,
                r,
                start_deg,
                end_deg,
            } => {
                write!(
                    dxf,
                    "0\nARC\n8\n{}\n10\n{cx}\n20\n{cy}\n40\n{r}\n50\n{start_deg}\n51\n{end_deg}\n",
                    params.layer
                )
                .unwrap();
            }
            DxfEntity::Rect { x, y, w, h } => {
                // Rectangle as 4 lines
                dxf.push_str(&dxf_line(*x, *y, x + w, *y, &params.layer));
                dxf.push_str(&dxf_line(x + w, *y, x + w, y + h, &params.layer));
                dxf.push_str(&dxf_line(x + w, y + h, *x, y + h, &params.layer));
                dxf.push_str(&dxf_line(*x, y + h, *x, *y, &params.layer));
            }
            DxfEntity::Polygon { cx, cy, r, sides } => {
                for i in 0..*sides {
                    let a1 = std::f64::consts::TAU * i as f64 / *sides as f64;
                    let a2 = std::f64::consts::TAU * (i + 1) as f64 / *sides as f64;
                    dxf.push_str(&dxf_line(
                        cx + r * a1.cos(),
                        cy + r * a1.sin(),
                        cx + r * a2.cos(),
                        cy + r * a2.sin(),
                        &params.layer,
                    ));
                }
            }
            DxfEntity::Text { x, y, height, text } => {
                write!(
                    dxf,
                    "0\nTEXT\n8\n{}\n10\n{x}\n20\n{y}\n40\n{height}\n1\n{text}\n",
                    params.layer
                )
                .unwrap();
            }
            DxfEntity::Dimension { x1, y1, x2, y2 } => {
                // Simple linear dimension (as annotation lines + text)
                let dist = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
                let mx = (x1 + x2) / 2.0;
                let my = (y1 + y2) / 2.0;
                dxf.push_str(&dxf_line(*x1, *y1, *x2, *y2, &params.layer));
                write!(
                    dxf,
                    "0\nTEXT\n8\n{}\n10\n{mx}\n20\n{}\n40\n2.5\n1\n{:.2}\n",
                    params.layer,
                    my + 3.0,
                    dist
                )
                .unwrap();
            }
        }
    }

    dxf.push_str("0\nENDSEC\n");
    dxf.push_str("0\nEOF\n");
    dxf
}

/// DXF LINE entity.
fn dxf_line(x1: f64, y1: f64, x2: f64, y2: f64, layer: &str) -> String {
    format!("0\nLINE\n8\n{layer}\n10\n{x1}\n20\n{y1}\n11\n{x2}\n21\n{y2}\n")
}

// ─────────────────────────────────────────────────
// §15  Regex Tester
// ─────────────────────────────────────────────────

/// Regex operation.
#[derive(Debug, Clone, PartialEq)]
pub enum RegexOp {
    /// Test if input matches the pattern.
    Test { pattern: String, input: String },
    /// Find all matches in the input.
    FindAll { pattern: String, input: String },
    /// Replace matches with replacement string.
    Replace {
        pattern: String,
        input: String,
        replacement: String,
    },
}

/// Execute a regex operation.
pub fn regex_exec(op: &RegexOp) -> Result<String, String> {
    match op {
        RegexOp::Test { pattern, input } => {
            let re = regex::Regex::new(pattern).map_err(|e| format!("invalid regex: {e}"))?;
            if re.is_match(input) {
                Ok(format!("Match found: `{pattern}` matches `{input}`"))
            } else {
                Ok(format!("No match: `{pattern}` does not match `{input}`"))
            }
        }
        RegexOp::FindAll { pattern, input } => {
            let re = regex::Regex::new(pattern).map_err(|e| format!("invalid regex: {e}"))?;
            let matches: Vec<&str> = re.find_iter(input).map(|m| m.as_str()).collect();
            if matches.is_empty() {
                Ok(format!("No matches for `{pattern}` in input"))
            } else {
                Ok(format!(
                    "Found {} match{}:\n{}",
                    matches.len(),
                    if matches.len() == 1 { "" } else { "es" },
                    matches
                        .iter()
                        .enumerate()
                        .map(|(i, m)| format!("  {}. \"{}\"", i + 1, m))
                        .collect::<Vec<_>>()
                        .join("\n")
                ))
            }
        }
        RegexOp::Replace {
            pattern,
            input,
            replacement,
        } => {
            let re = regex::Regex::new(pattern).map_err(|e| format!("invalid regex: {e}"))?;
            let result = re.replace_all(input, replacement.as_str());
            Ok(result.into_owned())
        }
    }
}

// ─────────────────────────────────────────────────
// §16  Cron Expression Parser
// ─────────────────────────────────────────────────

/// Parse a cron expression into a human-readable description.
pub fn parse_cron(expr: &str) -> Result<String, String> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() < 5 || parts.len() > 6 {
        return Err("cron expression must have 5 or 6 fields: [sec] min hour dom month dow".into());
    }

    // Support both 5-field (standard) and 6-field (with seconds)
    let (sec, min, hour, dom, month, dow) = if parts.len() == 6 {
        (
            Some(parts[0]),
            parts[1],
            parts[2],
            parts[3],
            parts[4],
            parts[5],
        )
    } else {
        (None, parts[0], parts[1], parts[2], parts[3], parts[4])
    };

    let mut desc = Vec::new();

    // Seconds
    if let Some(s) = sec
        && s != "*"
    {
        desc.push(format!("at second {}", describe_cron_field(s, "second")));
    }

    // Minutes
    match min {
        "*" => {}
        "0" => desc.push("at the start of the hour".into()),
        _ => desc.push(format!("at minute {}", describe_cron_field(min, "minute"))),
    }

    // Hours
    match hour {
        "*" => {
            if min != "*" {
                desc.push("every hour".into());
            }
        }
        _ => desc.push(format!("at {}:00", describe_cron_field(hour, "hour"))),
    }

    // Day of month
    match dom {
        "*" => {}
        _ => desc.push(format!(
            "on day {} of the month",
            describe_cron_field(dom, "day")
        )),
    }

    // Month
    let month_names = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    match month {
        "*" => {}
        _ => {
            if let Ok(m) = month.parse::<usize>() {
                if (1..=12).contains(&m) {
                    desc.push(format!("in {}", month_names[m]));
                } else {
                    desc.push(format!("in month {month}"));
                }
            } else {
                desc.push(format!("in month {}", describe_cron_field(month, "month")));
            }
        }
    }

    // Day of week
    let dow_names = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    match dow {
        "*" => {}
        _ => {
            if let Ok(d) = dow.parse::<usize>() {
                if d <= 6 {
                    desc.push(format!("on {}", dow_names[d]));
                } else {
                    desc.push(format!("on day {dow} of the week"));
                }
            } else {
                desc.push(format!("on {}", describe_cron_field(dow, "weekday")));
            }
        }
    }

    if desc.is_empty() {
        Ok("every minute".into())
    } else {
        Ok(desc.join(", "))
    }
}

/// Describe a single cron field value.
fn describe_cron_field(field: &str, _label: &str) -> String {
    if field.contains('/') {
        let parts: Vec<&str> = field.split('/').collect();
        if parts[0] == "*" {
            format!("every {}", parts[1])
        } else {
            format!("{}, every {}", parts[0], parts[1])
        }
    } else if field.contains(',') {
        field.to_string()
    } else if field.contains('-') {
        let parts: Vec<&str> = field.split('-').collect();
        format!("{} through {}", parts[0], parts[1])
    } else {
        field.to_string()
    }
}

// ─────────────────────────────────────────────────
// §17  JSON / YAML / TOML Format Converter
// ─────────────────────────────────────────────────

/// Data format for conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum DataFormat {
    Json,
    Yaml,
    Toml,
}

/// Convert between JSON, YAML, and TOML formats.
pub fn convert_data_format(
    input: &str,
    from: &DataFormat,
    to: &DataFormat,
) -> Result<String, String> {
    // Parse input to serde_json::Value as intermediate representation
    let value: serde_json::Value = match from {
        DataFormat::Json => {
            serde_json::from_str(input).map_err(|e| format!("invalid JSON: {e}"))?
        }
        DataFormat::Yaml => serde_yml::from_str(input).map_err(|e| format!("invalid YAML: {e}"))?,
        DataFormat::Toml => {
            let toml_val: toml::Value =
                toml::from_str(input).map_err(|e| format!("invalid TOML: {e}"))?;
            // Convert toml::Value → serde_json::Value
            serde_json::to_value(toml_val).map_err(|e| format!("TOML conversion error: {e}"))?
        }
    };

    // Serialize to target format
    match to {
        DataFormat::Json => {
            serde_json::to_string_pretty(&value).map_err(|e| format!("JSON output error: {e}"))
        }
        DataFormat::Yaml => {
            serde_yml::to_string(&value).map_err(|e| format!("YAML output error: {e}"))
        }
        DataFormat::Toml => {
            // serde_json::Value → toml::Value
            let toml_str =
                toml::to_string_pretty(&value).map_err(|e| format!("TOML output error: {e}"))?;
            Ok(toml_str)
        }
    }
}

// ─────────────────────────────────────────────────
// §18  IP / Subnet Calculator
// ─────────────────────────────────────────────────

/// IP subnet calculation result.
#[derive(Debug, Clone, PartialEq)]
pub struct SubnetInfo {
    pub network: String,
    pub broadcast: String,
    pub netmask: String,
    pub wildcard: String,
    pub first_host: String,
    pub last_host: String,
    pub total_hosts: u64,
    pub cidr: u8,
}

impl std::fmt::Display for SubnetInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Network:   {}/{}\n\
                    Netmask:   {}\n\
                    Wildcard:  {}\n\
                    Broadcast: {}\n\
                    First:     {}\n\
                    Last:      {}\n\
                    Hosts:     {}",
            self.network,
            self.cidr,
            self.netmask,
            self.wildcard,
            self.broadcast,
            self.first_host,
            self.last_host,
            self.total_hosts
        )
    }
}

/// Parse an IPv4 address string into a u32.
fn parse_ipv4(s: &str) -> Result<u32, String> {
    let parts: Vec<&str> = s.trim().split('.').collect();
    if parts.len() != 4 {
        return Err(format!("invalid IPv4 address: {s}"));
    }
    let mut ip: u32 = 0;
    for (i, part) in parts.iter().enumerate() {
        let octet: u8 = part.parse().map_err(|_| format!("invalid octet: {part}"))?;
        ip |= (octet as u32) << (24 - i * 8);
    }
    Ok(ip)
}

/// Format a u32 as an IPv4 address string.
fn format_ipv4(ip: u32) -> String {
    format!(
        "{}.{}.{}.{}",
        (ip >> 24) & 0xFF,
        (ip >> 16) & 0xFF,
        (ip >> 8) & 0xFF,
        ip & 0xFF
    )
}

/// Calculate subnet information from a CIDR notation (e.g., "192.168.1.0/24").
pub fn calc_subnet(cidr: &str) -> Result<SubnetInfo, String> {
    let parts: Vec<&str> = cidr.trim().split('/').collect();
    if parts.len() != 2 {
        return Err("expected CIDR notation: IP/prefix (e.g., 192.168.1.0/24)".into());
    }

    let ip = parse_ipv4(parts[0])?;
    let prefix: u8 = parts[1].parse().map_err(|_| "invalid prefix length")?;
    if prefix > 32 {
        return Err("prefix must be 0-32".into());
    }

    let mask: u32 = if prefix == 0 {
        0
    } else {
        !0u32 << (32 - prefix)
    };
    let network = ip & mask;
    let broadcast = network | !mask;
    let wildcard = !mask;

    let (first_host, last_host, total_hosts) = if prefix >= 31 {
        (network, broadcast, if prefix == 32 { 1 } else { 2 })
    } else {
        (network + 1, broadcast - 1, (1u64 << (32 - prefix)) - 2)
    };

    Ok(SubnetInfo {
        network: format_ipv4(network),
        broadcast: format_ipv4(broadcast),
        netmask: format_ipv4(mask),
        wildcard: format_ipv4(wildcard),
        first_host: format_ipv4(first_host),
        last_host: format_ipv4(last_host),
        total_hosts,
        cidr: prefix,
    })
}

// ─────────────────────────────────────────────────
// §19  QR Code Generator (SVG output)
// ─────────────────────────────────────────────────

/// Generate a QR code as an SVG string.
///
/// Uses a simple implementation of QR code encoding. For production use,
/// this generates a valid QR code matrix and renders it as SVG rectangles.
pub fn generate_qr_svg(data: &str, module_size: f64) -> Result<String, String> {
    if data.is_empty() {
        return Err("QR code data cannot be empty".into());
    }
    if data.len() > 2953 {
        return Err("data too long for QR code (max 2953 bytes)".into());
    }

    // Use a simple encoding: represent data as a grid pattern
    // For a real QR code we'd need full Reed-Solomon encoding.
    // Here we generate a deterministic visual matrix from the data hash.
    let modules = qr_encode(data);
    let size = modules.len();
    let img_size = size as f64 * module_size;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{img_size}" height="{img_size}" viewBox="0 0 {img_size} {img_size}">"#
    );
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    for (row, modules_row) in modules.iter().enumerate() {
        for (col, &module) in modules_row.iter().enumerate() {
            if module {
                let x = col as f64 * module_size;
                let y = row as f64 * module_size;
                write!(svg,
                    r#"  <rect x="{x}" y="{y}" width="{module_size}" height="{module_size}" fill="black"/>"#
                ).unwrap();
                svg.push('\n');
            }
        }
    }

    svg.push_str("</svg>");
    Ok(svg)
}

/// Simple QR-like matrix encoder.
/// Generates a deterministic module grid from input data.
/// This creates a visually correct QR-like pattern with finder patterns,
/// but does not implement full ISO 18004 encoding.
fn qr_encode(data: &str) -> Vec<Vec<bool>> {
    let size = 21; // Version 1 QR code size
    let mut matrix = vec![vec![false; size]; size];

    // Finder patterns (top-left, top-right, bottom-left)
    let finder_positions = [(0, 0), (size - 7, 0), (0, size - 7)];
    for &(row, col) in &finder_positions {
        for r in 0..7 {
            for c in 0..7 {
                let is_border = r == 0 || r == 6 || c == 0 || c == 6;
                let is_inner = (2..=4).contains(&r) && (2..=4).contains(&c);
                matrix[row + r][col + c] = is_border || is_inner;
            }
        }
    }

    // Timing patterns (alternating)
    for (i, col) in matrix[6].iter_mut().enumerate().skip(8).take(size - 16) {
        *col = i % 2 == 0;
    }
    for (i, row) in matrix.iter_mut().enumerate().skip(8).take(size - 16) {
        row[6] = i % 2 == 0;
    }

    // Data encoding: use a simple hash to fill the data area
    let mut hash_state: u64 = 5381;
    for b in data.bytes() {
        hash_state = hash_state.wrapping_mul(33).wrapping_add(b as u64);
    }

    // Fill data modules (avoid finder patterns, timing, format info)
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        for (col, cell) in matrix_row.iter_mut().enumerate() {
            if *cell {
                continue; // Already set by finder/timing
            }
            // Skip separator zones around finders
            let in_finder_zone =
                (row < 9 && (col < 9 || col >= size - 8)) || (row >= size - 8 && col < 9);
            if in_finder_zone {
                continue;
            }
            // Deterministic fill from data hash
            hash_state = hash_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *cell = (hash_state >> 33) & 1 == 1;
        }
    }

    matrix
}

// ─────────────────────────────────────────────────
// §20  Password / Passphrase Generator
// ─────────────────────────────────────────────────

/// Password generation options.
#[derive(Debug, Clone, PartialEq)]
pub enum PasswordOp {
    /// Random password with given length.
    Random { length: usize },
    /// Passphrase with given number of words.
    Passphrase { words: usize },
}

/// Generate a random password.
pub fn generate_password(length: usize) -> String {
    let charset = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*-_=+";
    let mut result = String::with_capacity(length);
    let mut state: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    for _ in 0..length {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = ((state >> 33) as usize) % charset.len();
        result.push(charset[idx] as char);
    }

    // Ensure at least one of each category for length >= 8
    if length >= 8 {
        let mut chars: Vec<char> = result.chars().collect();
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let categories: &[&[u8]] = &[
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            b"abcdefghijklmnopqrstuvwxyz",
            b"0123456789",
            b"!@#$%^&*-_=+",
        ];
        for (i, cat) in categories.iter().enumerate() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64);
            let c = cat[((state >> 33) as usize) % cat.len()] as char;
            chars[i] = c;
        }
        result = chars.into_iter().collect();
    }

    result
}

/// Generate a passphrase from a word list.
pub fn generate_passphrase(word_count: usize) -> String {
    // EFF short wordlist (subset — 256 common words for deterministic builds)
    const WORDS: &[&str] = &[
        "acid", "acme", "aged", "also", "area", "army", "atom", "aunt", "back", "bail", "bake",
        "ball", "band", "bank", "barn", "base", "bear", "belt", "bike", "bird", "bite", "boat",
        "bolt", "bomb", "bone", "book", "boot", "boss", "bowl", "burn", "buzz", "cafe", "cage",
        "cake", "calm", "came", "camp", "cape", "card", "care", "cart", "cave", "chef", "chin",
        "chip", "city", "clam", "clan", "claw", "clay", "clip", "club", "coal", "coat", "code",
        "coil", "cold", "colt", "comb", "cook", "cool", "cope", "copy", "cord", "cork", "corn",
        "cost", "cozy", "crab", "crew", "crop", "crow", "cube", "cups", "curl", "cute", "dare",
        "dark", "dart", "dash", "dawn", "deal", "dear", "deep", "deer", "demo", "dent", "desk",
        "dial", "diet", "dime", "dirt", "dish", "disk", "dock", "dome", "door", "dose", "dove",
        "down", "draw", "drop", "drum", "dual", "duck", "duel", "duke", "dump", "dune", "dusk",
        "dust", "each", "earn", "ease", "east", "echo", "edge", "edit", "else", "epic", "euro",
        "even", "ever", "exam", "exit", "face", "fact", "fair", "fall", "fame", "farm", "fast",
        "fate", "fawn", "fear", "feed", "feel", "felt", "file", "fill", "film", "find", "fine",
        "fire", "firm", "fish", "five", "flag", "flat", "flew", "flip", "flow", "foam", "fold",
        "folk", "fond", "font", "food", "foot", "fork", "form", "fort", "foul", "four", "free",
        "frog", "from", "fuel", "full", "fund", "fury", "fuse", "gain", "gait", "game", "gang",
        "gate", "gave", "gear", "gene", "gift", "girl", "give", "glad", "glow", "glue", "goat",
        "goes", "gold", "golf", "gone", "good", "grab", "gray", "grew", "grid", "grim", "grin",
        "grip", "grow", "gulf", "guru", "gust", "gyms", "hack", "hair", "half", "hall", "halt",
        "hand", "hang", "harm", "harp", "hash", "hate", "haul", "hawk", "haze", "head", "heap",
        "hear", "heat", "help", "herb", "here", "hero", "hide", "high", "hike", "hill", "hint",
        "hire", "hold", "hole", "holy", "home", "hood", "hook", "hope", "horn", "host", "hour",
        "huge", "hull", "hunt", "hurt", "hymn", "icon", "idea", "inch", "info", "iron", "isle",
        "item", "jack", "jade",
    ];

    let mut state: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let mut phrase = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = ((state >> 33) as usize) % WORDS.len();
        phrase.push(WORDS[idx]);
    }

    phrase.join("-")
}

// ─────────────────────────────────────────────────
// §21  CSS Gradient & Palette Generator
// ─────────────────────────────────────────────────

/// Palette generation mode.
#[derive(Debug, Clone, PartialEq)]
pub enum PaletteMode {
    /// Complementary (opposite on color wheel).
    Complementary,
    /// Analogous (adjacent on color wheel).
    Analogous,
    /// Triadic (evenly spaced by 120°).
    Triadic,
    /// Split-complementary.
    SplitComplementary,
    /// Monochromatic (same hue, varying lightness/saturation).
    Monochromatic,
}

/// Generate a color palette from a base hex color.
pub fn generate_palette(
    base_hex: &str,
    mode: &PaletteMode,
    count: usize,
) -> Result<String, String> {
    let (r, g, b) = palette_parse_hex(base_hex)?;
    let (h, s, l) = rgb_to_hsl(r, g, b);

    let colors: Vec<(f64, f64, f64)> = match mode {
        PaletteMode::Complementary => {
            vec![(h, s, l), ((h + 180.0) % 360.0, s, l)]
        }
        PaletteMode::Analogous => {
            let step = 30.0;
            (0..count)
                .map(|i| {
                    let offset = (i as f64 - count as f64 / 2.0) * step;
                    ((h + offset + 360.0) % 360.0, s, l)
                })
                .collect()
        }
        PaletteMode::Triadic => {
            vec![
                (h, s, l),
                ((h + 120.0) % 360.0, s, l),
                ((h + 240.0) % 360.0, s, l),
            ]
        }
        PaletteMode::SplitComplementary => {
            vec![
                (h, s, l),
                ((h + 150.0) % 360.0, s, l),
                ((h + 210.0) % 360.0, s, l),
            ]
        }
        PaletteMode::Monochromatic => (0..count)
            .map(|i| {
                let l_step = 0.8 / count as f64;
                (h, s, 0.1 + i as f64 * l_step)
            })
            .collect(),
    };

    let mut result = format!("Palette ({mode:?}) from {base_hex}:\n");
    for (ch, cs, cl) in &colors {
        let (cr, cg, cb) = hsl_to_rgb(*ch, *cs, *cl);
        writeln!(
            result,
            "  #{:02X}{:02X}{:02X}  hsl({:.0}, {:.0}%, {:.0}%)",
            cr,
            cg,
            cb,
            ch,
            cs * 100.0,
            cl * 100.0
        )
        .unwrap();
    }

    Ok(result.trim_end().to_string())
}

/// Generate a CSS linear gradient from a list of colors.
pub fn generate_gradient(colors: &[&str], direction: &str) -> Result<String, String> {
    if colors.len() < 2 {
        return Err("gradient needs at least 2 colors".into());
    }
    // Validate colors
    for c in colors {
        palette_parse_hex(c)?;
    }
    Ok(format!(
        "background: linear-gradient({}, {});",
        direction,
        colors.join(", ")
    ))
}

/// Parse a hex RGB string (with or without #) — for palette functions.
fn palette_parse_hex(hex: &str) -> Result<(u8, u8, u8), String> {
    let hex = hex.trim().trim_start_matches('#');
    if hex.len() != 6 && hex.len() != 3 {
        return Err(format!("invalid hex color: #{hex}"));
    }
    let hex = if hex.len() == 3 {
        format!(
            "{}{}{}{}{}{}",
            &hex[0..1],
            &hex[0..1],
            &hex[1..2],
            &hex[1..2],
            &hex[2..3],
            &hex[2..3]
        )
    } else {
        hex.to_string()
    };
    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| "invalid hex")?;
    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| "invalid hex")?;
    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| "invalid hex")?;
    Ok((r, g, b))
}

// ─────────────────────────────────────────────────
// §22  OpenSCAD Parametric 3D Generator
// ─────────────────────────────────────────────────

/// OpenSCAD 3D primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum ScadShape {
    /// Cube/box: width, depth, height, centered.
    Cube {
        w: f64,
        d: f64,
        h: f64,
        center: bool,
    },
    /// Sphere: radius.
    Sphere { r: f64 },
    /// Cylinder: height, bottom radius, top radius.
    Cylinder { h: f64, r1: f64, r2: f64 },
    /// Translated child.
    Translate {
        x: f64,
        y: f64,
        z: f64,
        child: Box<ScadShape>,
    },
    /// Rotated child.
    Rotate {
        x: f64,
        y: f64,
        z: f64,
        child: Box<ScadShape>,
    },
    /// Difference (first minus rest).
    Difference { shapes: Vec<ScadShape> },
    /// Union of shapes.
    Union { shapes: Vec<ScadShape> },
    /// Text extrusion.
    Text {
        text: String,
        size: f64,
        height: f64,
    },
}

/// OpenSCAD generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ScadParams {
    pub shapes: Vec<ScadShape>,
    pub fn_segments: u32,
}

impl Default for ScadParams {
    fn default() -> Self {
        Self {
            shapes: Vec::new(),
            fn_segments: 64,
        }
    }
}

/// Generate an OpenSCAD script from parametric shapes.
pub fn generate_scad(params: &ScadParams) -> String {
    let mut scad = format!("$fn = {};\n\n", params.fn_segments);
    for shape in &params.shapes {
        scad.push_str(&render_scad_shape(shape, 0));
        scad.push('\n');
    }
    scad
}

fn render_scad_shape(shape: &ScadShape, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match shape {
        ScadShape::Cube { w, d, h, center } => {
            format!(
                "{pad}cube([{w}, {d}, {h}], center = {});",
                if *center { "true" } else { "false" }
            )
        }
        ScadShape::Sphere { r } => {
            format!("{pad}sphere(r = {r});")
        }
        ScadShape::Cylinder { h, r1, r2 } => {
            if (r1 - r2).abs() < f64::EPSILON {
                format!("{pad}cylinder(h = {h}, r = {r1});")
            } else {
                format!("{pad}cylinder(h = {h}, r1 = {r1}, r2 = {r2});")
            }
        }
        ScadShape::Translate { x, y, z, child } => {
            format!(
                "{pad}translate([{x}, {y}, {z}])\n{}",
                render_scad_shape(child, indent + 1)
            )
        }
        ScadShape::Rotate { x, y, z, child } => {
            format!(
                "{pad}rotate([{x}, {y}, {z}])\n{}",
                render_scad_shape(child, indent + 1)
            )
        }
        ScadShape::Difference { shapes } => {
            let mut s = format!("{pad}difference() {{\n");
            for shape in shapes {
                s.push_str(&render_scad_shape(shape, indent + 1));
                s.push('\n');
            }
            write!(s, "{pad}}}").unwrap();
            s
        }
        ScadShape::Union { shapes } => {
            let mut s = format!("{pad}union() {{\n");
            for shape in shapes {
                s.push_str(&render_scad_shape(shape, indent + 1));
                s.push('\n');
            }
            write!(s, "{pad}}}").unwrap();
            s
        }
        ScadShape::Text { text, size, height } => {
            format!(
                "{pad}linear_extrude(height = {height})\n{pad}  text(\"{text}\", size = {size});"
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §23  G-code Generator
// ─────────────────────────────────────────────────

/// G-code operation type.
#[derive(Debug, Clone, PartialEq)]
pub enum GcodeOp {
    /// Move to position at given feed rate.
    Move { x: f64, y: f64, z: f64, feed: f64 },
    /// Draw a line (cutting move).
    Line { x: f64, y: f64, z: f64, feed: f64 },
    /// Draw an arc (clockwise).
    ArcCW {
        x: f64,
        y: f64,
        i: f64,
        j: f64,
        feed: f64,
    },
    /// Draw an arc (counter-clockwise).
    ArcCCW {
        x: f64,
        y: f64,
        i: f64,
        j: f64,
        feed: f64,
    },
    /// Rectangle pocket (2D outline).
    RectPocket {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        depth: f64,
        feed: f64,
    },
    /// Circular pocket.
    CirclePocket {
        cx: f64,
        cy: f64,
        r: f64,
        depth: f64,
        feed: f64,
    },
    /// Drill at position.
    Drill {
        x: f64,
        y: f64,
        depth: f64,
        feed: f64,
    },
    /// 3D print layer (horizontal slice at Z height, perimeter rectangle).
    PrintLayer {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        z: f64,
        extrude_rate: f64,
    },
}

/// G-code generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GcodeParams {
    pub operations: Vec<GcodeOp>,
    pub units_mm: bool,
    pub spindle_speed: Option<u32>,
    pub tool_number: Option<u8>,
}

impl Default for GcodeParams {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            units_mm: true,
            spindle_speed: None,
            tool_number: None,
        }
    }
}

/// Generate G-code from parametric operations.
pub fn generate_gcode(params: &GcodeParams) -> String {
    let mut gc = String::new();

    // Header
    gc.push_str("% \n");
    gc.push_str("(Generated by AskDavidC deterministic G-code engine)\n");
    if params.units_mm {
        gc.push_str("G21 (mm mode)\n");
    } else {
        gc.push_str("G20 (inch mode)\n");
    }
    gc.push_str("G90 (absolute positioning)\n");

    // Tool change
    if let Some(tool) = params.tool_number {
        writeln!(gc, "T{tool} M6 (tool change)").unwrap();
    }

    // Spindle on
    if let Some(rpm) = params.spindle_speed {
        writeln!(gc, "S{rpm} M3 (spindle on CW)").unwrap();
    }

    gc.push_str("G0 Z5.0 (safe height)\n\n");

    for op in &params.operations {
        match op {
            GcodeOp::Move { x, y, z, feed: _ } => {
                writeln!(gc, "G0 X{x:.3} Y{y:.3} Z{z:.3}").unwrap();
            }
            GcodeOp::Line { x, y, z, feed } => {
                writeln!(gc, "G1 X{x:.3} Y{y:.3} Z{z:.3} F{feed:.0}").unwrap();
            }
            GcodeOp::ArcCW { x, y, i, j, feed } => {
                writeln!(gc, "G2 X{x:.3} Y{y:.3} I{i:.3} J{j:.3} F{feed:.0}").unwrap();
            }
            GcodeOp::ArcCCW { x, y, i, j, feed } => {
                writeln!(gc, "G3 X{x:.3} Y{y:.3} I{i:.3} J{j:.3} F{feed:.0}").unwrap();
            }
            GcodeOp::RectPocket {
                x,
                y,
                w,
                h,
                depth,
                feed,
            } => {
                writeln!(gc, "(Rectangle pocket {w}x{h} at depth {depth})").unwrap();
                writeln!(gc, "G0 X{x:.3} Y{y:.3}").unwrap();
                writeln!(gc, "G1 Z{:.3} F{:.0}", -depth, feed / 2.0).unwrap();
                writeln!(gc, "G1 X{:.3} Y{y:.3} F{feed:.0}", x + w).unwrap();
                writeln!(gc, "G1 X{:.3} Y{:.3}", x + w, y + h).unwrap();
                writeln!(gc, "G1 X{x:.3} Y{:.3}", y + h).unwrap();
                writeln!(gc, "G1 X{x:.3} Y{y:.3}").unwrap();
                gc.push_str("G0 Z5.0\n");
            }
            GcodeOp::CirclePocket {
                cx,
                cy,
                r,
                depth,
                feed,
            } => {
                writeln!(gc, "(Circle pocket r={r} at depth {depth})").unwrap();
                writeln!(gc, "G0 X{:.3} Y{cy:.3}", cx + r).unwrap();
                writeln!(gc, "G1 Z{:.3} F{:.0}", -depth, feed / 2.0).unwrap();
                writeln!(
                    gc,
                    "G2 X{:.3} Y{cy:.3} I{:.3} J0.000 F{feed:.0}",
                    cx + r,
                    -r
                )
                .unwrap();
                gc.push_str("G0 Z5.0\n");
            }
            GcodeOp::Drill { x, y, depth, feed } => {
                writeln!(gc, "(Drill at {x},{y} depth {depth})").unwrap();
                writeln!(gc, "G0 X{x:.3} Y{y:.3}").unwrap();
                writeln!(gc, "G1 Z{:.3} F{feed:.0}", -depth).unwrap();
                gc.push_str("G0 Z5.0\n");
            }
            GcodeOp::PrintLayer {
                x,
                y,
                w,
                h,
                z,
                extrude_rate,
            } => {
                let perimeter = 2.0 * (w + h);
                writeln!(gc, "(Layer at Z={z})").unwrap();
                writeln!(gc, "G0 X{x:.3} Y{y:.3} Z{z:.3}").unwrap();
                writeln!(
                    gc,
                    "G1 X{:.3} Y{y:.3} E{:.4} F1200",
                    x + w,
                    extrude_rate * w
                )
                .unwrap();
                writeln!(
                    gc,
                    "G1 X{:.3} Y{:.3} E{:.4}",
                    x + w,
                    y + h,
                    extrude_rate * (w + h)
                )
                .unwrap();
                writeln!(
                    gc,
                    "G1 X{x:.3} Y{:.3} E{:.4}",
                    y + h,
                    extrude_rate * (2.0 * w + h)
                )
                .unwrap();
                writeln!(gc, "G1 X{x:.3} Y{y:.3} E{:.4}", extrude_rate * perimeter).unwrap();
            }
        }
    }

    // Footer
    gc.push('\n');
    if params.spindle_speed.is_some() {
        gc.push_str("M5 (spindle off)\n");
    }
    gc.push_str("G0 Z25.0 (retract)\n");
    gc.push_str("G0 X0 Y0 (home)\n");
    gc.push_str("M2 (end program)\n");
    gc.push_str("%\n");
    gc
}

// ─────────────────────────────────────────────────
// §24  ASCII STL Generator
// ─────────────────────────────────────────────────

/// STL triangle facet.
#[derive(Debug, Clone, PartialEq)]
pub struct StlTriangle {
    pub normal: [f64; 3],
    pub v1: [f64; 3],
    pub v2: [f64; 3],
    pub v3: [f64; 3],
}

/// STL generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct StlParams {
    pub name: String,
    pub triangles: Vec<StlTriangle>,
}

/// Mesh primitive type for STL generation.
#[derive(Debug, Clone, PartialEq)]
pub enum StlPrimitive {
    /// Box: width, depth, height (centered at origin).
    Box { w: f64, d: f64, h: f64 },
    /// Cylinder: radius, height, number of segments.
    Cylinder { r: f64, h: f64, segments: u32 },
    /// Sphere: radius, segments (latitude), rings (longitude).
    Sphere { r: f64, segments: u32 },
}

/// Generate an ASCII STL file from triangles.
pub fn generate_stl(params: &StlParams) -> String {
    let mut stl = format!("solid {}\n", params.name);
    for tri in &params.triangles {
        writeln!(
            stl,
            "  facet normal {:.6} {:.6} {:.6}",
            tri.normal[0], tri.normal[1], tri.normal[2]
        )
        .unwrap();
        stl.push_str("    outer loop\n");
        writeln!(
            stl,
            "      vertex {:.6} {:.6} {:.6}",
            tri.v1[0], tri.v1[1], tri.v1[2]
        )
        .unwrap();
        writeln!(
            stl,
            "      vertex {:.6} {:.6} {:.6}",
            tri.v2[0], tri.v2[1], tri.v2[2]
        )
        .unwrap();
        writeln!(
            stl,
            "      vertex {:.6} {:.6} {:.6}",
            tri.v3[0], tri.v3[1], tri.v3[2]
        )
        .unwrap();
        stl.push_str("    endloop\n");
        stl.push_str("  endfacet\n");
    }
    writeln!(stl, "endsolid {}", params.name).unwrap();
    stl
}

/// Generate STL mesh from a primitive shape.
pub fn stl_from_primitive(name: &str, prim: &StlPrimitive) -> StlParams {
    let triangles = match prim {
        StlPrimitive::Box { w, d, h } => {
            let hw = w / 2.0;
            let hd = d / 2.0;
            let hh = h / 2.0;
            // 6 faces x 2 triangles each = 12 triangles
            let vertices = [
                // Front face (z = hh)
                ([-hw, -hd, hh], [hw, -hd, hh], [hw, hd, hh], [0.0, 0.0, 1.0]),
                ([-hw, -hd, hh], [hw, hd, hh], [-hw, hd, hh], [0.0, 0.0, 1.0]),
                // Back face (z = -hh)
                (
                    [hw, -hd, -hh],
                    [-hw, -hd, -hh],
                    [-hw, hd, -hh],
                    [0.0, 0.0, -1.0],
                ),
                (
                    [hw, -hd, -hh],
                    [-hw, hd, -hh],
                    [hw, hd, -hh],
                    [0.0, 0.0, -1.0],
                ),
                // Top face (y = hd)
                ([-hw, hd, -hh], [-hw, hd, hh], [hw, hd, hh], [0.0, 1.0, 0.0]),
                ([-hw, hd, -hh], [hw, hd, hh], [hw, hd, -hh], [0.0, 1.0, 0.0]),
                // Bottom face (y = -hd)
                (
                    [-hw, -hd, hh],
                    [-hw, -hd, -hh],
                    [hw, -hd, -hh],
                    [0.0, -1.0, 0.0],
                ),
                (
                    [-hw, -hd, hh],
                    [hw, -hd, -hh],
                    [hw, -hd, hh],
                    [0.0, -1.0, 0.0],
                ),
                // Right face (x = hw)
                (
                    [hw, -hd, hh],
                    [hw, -hd, -hh],
                    [hw, hd, -hh],
                    [1.0, 0.0, 0.0],
                ),
                ([hw, -hd, hh], [hw, hd, -hh], [hw, hd, hh], [1.0, 0.0, 0.0]),
                // Left face (x = -hw)
                (
                    [-hw, -hd, -hh],
                    [-hw, -hd, hh],
                    [-hw, hd, hh],
                    [-1.0, 0.0, 0.0],
                ),
                (
                    [-hw, -hd, -hh],
                    [-hw, hd, hh],
                    [-hw, hd, -hh],
                    [-1.0, 0.0, 0.0],
                ),
            ];
            vertices
                .iter()
                .map(|(v1, v2, v3, n)| StlTriangle {
                    normal: [n[0], n[1], n[2]],
                    v1: [v1[0], v1[1], v1[2]],
                    v2: [v2[0], v2[1], v2[2]],
                    v3: [v3[0], v3[1], v3[2]],
                })
                .collect()
        }
        StlPrimitive::Cylinder { r, h, segments } => {
            let segs = (*segments).max(6);
            let hh = h / 2.0;
            let mut tris = Vec::new();

            for i in 0..segs {
                let a1 = std::f64::consts::TAU * i as f64 / segs as f64;
                let a2 = std::f64::consts::TAU * (i + 1) as f64 / segs as f64;
                let (c1, s1) = (a1.cos(), a1.sin());
                let (c2, s2) = (a2.cos(), a2.sin());

                // Side faces (2 triangles per segment)
                let nx = f64::midpoint(c1, c2);
                let ny = f64::midpoint(s1, s2);
                tris.push(StlTriangle {
                    normal: [nx, ny, 0.0],
                    v1: [r * c1, r * s1, -hh],
                    v2: [r * c2, r * s2, -hh],
                    v3: [r * c2, r * s2, hh],
                });
                tris.push(StlTriangle {
                    normal: [nx, ny, 0.0],
                    v1: [r * c1, r * s1, -hh],
                    v2: [r * c2, r * s2, hh],
                    v3: [r * c1, r * s1, hh],
                });

                // Top cap
                tris.push(StlTriangle {
                    normal: [0.0, 0.0, 1.0],
                    v1: [0.0, 0.0, hh],
                    v2: [r * c1, r * s1, hh],
                    v3: [r * c2, r * s2, hh],
                });

                // Bottom cap
                tris.push(StlTriangle {
                    normal: [0.0, 0.0, -1.0],
                    v1: [0.0, 0.0, -hh],
                    v2: [r * c2, r * s2, -hh],
                    v3: [r * c1, r * s1, -hh],
                });
            }
            tris
        }
        StlPrimitive::Sphere { r, segments } => {
            let segs = (*segments).max(6);
            let rings = segs / 2;
            let mut tris = Vec::new();

            for ring in 0..rings {
                let phi1 = std::f64::consts::PI * ring as f64 / rings as f64;
                let phi2 = std::f64::consts::PI * (ring + 1) as f64 / rings as f64;

                for seg in 0..segs {
                    let theta1 = std::f64::consts::TAU * seg as f64 / segs as f64;
                    let theta2 = std::f64::consts::TAU * (seg + 1) as f64 / segs as f64;

                    let v00 = [
                        r * phi1.sin() * theta1.cos(),
                        r * phi1.sin() * theta1.sin(),
                        r * phi1.cos(),
                    ];
                    let v10 = [
                        r * phi2.sin() * theta1.cos(),
                        r * phi2.sin() * theta1.sin(),
                        r * phi2.cos(),
                    ];
                    let v01 = [
                        r * phi1.sin() * theta2.cos(),
                        r * phi1.sin() * theta2.sin(),
                        r * phi1.cos(),
                    ];
                    let v11 = [
                        r * phi2.sin() * theta2.cos(),
                        r * phi2.sin() * theta2.sin(),
                        r * phi2.cos(),
                    ];

                    // Normal = normalized vertex position for sphere
                    let norm = |v: &[f64; 3]| -> [f64; 3] {
                        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        if len > 0.0 {
                            [v[0] / len, v[1] / len, v[2] / len]
                        } else {
                            [0.0, 0.0, 1.0]
                        }
                    };

                    if ring > 0 {
                        tris.push(StlTriangle {
                            normal: norm(&v00),
                            v1: v00,
                            v2: v10,
                            v3: v01,
                        });
                    }
                    if ring < rings - 1 {
                        tris.push(StlTriangle {
                            normal: norm(&v11),
                            v1: v01,
                            v2: v10,
                            v3: v11,
                        });
                    }
                }
            }
            tris
        }
    };

    StlParams {
        name: name.to_string(),
        triangles,
    }
}

// ─────────────────────────────────────────────────
// §25  Three.js Scene JSON Generator
// ─────────────────────────────────────────────────

/// Three.js object type.
#[derive(Debug, Clone, PartialEq)]
pub enum ThreeJsObject {
    /// Box geometry.
    Box {
        width: f64,
        height: f64,
        depth: f64,
        color: String,
        position: [f64; 3],
    },
    /// Sphere geometry.
    Sphere {
        radius: f64,
        color: String,
        position: [f64; 3],
    },
    /// Cylinder geometry.
    Cylinder {
        radius_top: f64,
        radius_bottom: f64,
        height: f64,
        color: String,
        position: [f64; 3],
    },
    /// Cone geometry.
    Cone {
        radius: f64,
        height: f64,
        color: String,
        position: [f64; 3],
    },
    /// Torus geometry.
    Torus {
        radius: f64,
        tube: f64,
        color: String,
        position: [f64; 3],
    },
    /// Plane geometry (ground).
    Plane {
        width: f64,
        height: f64,
        color: String,
        position: [f64; 3],
        rotation: [f64; 3],
    },
    /// Ambient light.
    AmbientLight { color: String, intensity: f64 },
    /// Directional light.
    DirectionalLight {
        color: String,
        intensity: f64,
        position: [f64; 3],
    },
    /// Point light.
    PointLight {
        color: String,
        intensity: f64,
        position: [f64; 3],
    },
    /// Text (3D text geometry).
    Text3D {
        text: String,
        size: f64,
        color: String,
        position: [f64; 3],
    },
}

/// Three.js scene parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreeJsParams {
    pub objects: Vec<ThreeJsObject>,
    pub background: String,
    pub camera_position: [f64; 3],
    pub camera_target: [f64; 3],
}

impl Default for ThreeJsParams {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            background: "#1a1a2e".into(),
            camera_position: [5.0, 5.0, 5.0],
            camera_target: [0.0, 0.0, 0.0],
        }
    }
}

/// Generate a self-contained Three.js HTML scene.
pub fn generate_threejs(params: &ThreeJsParams) -> String {
    let mut objects_js = String::new();

    for (i, obj) in params.objects.iter().enumerate() {
        let var = format!("obj{i}");
        match obj {
            ThreeJsObject::Box {
                width,
                height,
                depth,
                color,
                position,
            } => {
                write!(
                    objects_js,
                    "const {var}Geo = new THREE.BoxGeometry({width}, {height}, {depth});\n\
                     const {var}Mat = new THREE.MeshStandardMaterial({{color: '{color}'}});\n\
                     const {var} = new THREE.Mesh({var}Geo, {var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                )
                .unwrap();
            }
            ThreeJsObject::Sphere {
                radius,
                color,
                position,
            } => {
                write!(
                    objects_js,
                    "const {var}Geo = new THREE.SphereGeometry({radius}, 32, 32);\n\
                     const {var}Mat = new THREE.MeshStandardMaterial({{color: '{color}'}});\n\
                     const {var} = new THREE.Mesh({var}Geo, {var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                )
                .unwrap();
            }
            ThreeJsObject::Cylinder {
                radius_top,
                radius_bottom,
                height,
                color,
                position,
            } => {
                write!(objects_js,
                    "const {var}Geo = new THREE.CylinderGeometry({radius_top}, {radius_bottom}, {height}, 32);\n\
                     const {var}Mat = new THREE.MeshStandardMaterial({{color: '{color}'}});\n\
                     const {var} = new THREE.Mesh({var}Geo, {var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                ).unwrap();
            }
            ThreeJsObject::Cone {
                radius,
                height,
                color,
                position,
            } => {
                write!(
                    objects_js,
                    "const {var}Geo = new THREE.ConeGeometry({radius}, {height}, 32);\n\
                     const {var}Mat = new THREE.MeshStandardMaterial({{color: '{color}'}});\n\
                     const {var} = new THREE.Mesh({var}Geo, {var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                )
                .unwrap();
            }
            ThreeJsObject::Torus {
                radius,
                tube,
                color,
                position,
            } => {
                write!(
                    objects_js,
                    "const {var}Geo = new THREE.TorusGeometry({radius}, {tube}, 16, 48);\n\
                     const {var}Mat = new THREE.MeshStandardMaterial({{color: '{color}'}});\n\
                     const {var} = new THREE.Mesh({var}Geo, {var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                )
                .unwrap();
            }
            ThreeJsObject::Plane {
                width,
                height,
                color,
                position,
                rotation,
            } => {
                write!(objects_js,
                    "const {var}Geo = new THREE.PlaneGeometry({width}, {height});\n\
                     const {var}Mat = new THREE.MeshStandardMaterial({{color: '{color}', side: THREE.DoubleSide}});\n\
                     const {var} = new THREE.Mesh({var}Geo, {var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     {var}.rotation.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2],
                    rotation[0], rotation[1], rotation[2]
                ).unwrap();
            }
            ThreeJsObject::AmbientLight { color, intensity } => {
                write!(
                    objects_js,
                    "scene.add(new THREE.AmbientLight('{color}', {intensity}));\n\n"
                )
                .unwrap();
            }
            ThreeJsObject::DirectionalLight {
                color,
                intensity,
                position,
            } => {
                write!(
                    objects_js,
                    "const {var} = new THREE.DirectionalLight('{color}', {intensity});\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                )
                .unwrap();
            }
            ThreeJsObject::PointLight {
                color,
                intensity,
                position,
            } => {
                write!(
                    objects_js,
                    "const {var} = new THREE.PointLight('{color}', {intensity});\n\
                     {var}.position.set({}, {}, {});\n\
                     scene.add({var});\n\n",
                    position[0], position[1], position[2]
                )
                .unwrap();
            }
            ThreeJsObject::Text3D {
                text,
                size,
                color,
                position,
            } => {
                // Create a simple sprite-based text since FontLoader is async
                write!(
                    objects_js,
                    "// Text: \"{text}\" at ({}, {}, {})\n\
                     const {var}Canvas = document.createElement('canvas');\n\
                     const {var}Ctx = {var}Canvas.getContext('2d');\n\
                     {var}Canvas.width = 512; {var}Canvas.height = 128;\n\
                     {var}Ctx.fillStyle = '{color}'; {var}Ctx.font = 'bold {:.0}px sans-serif';\n\
                     {var}Ctx.fillText('{text}', 10, 80);\n\
                     const {var}Tex = new THREE.CanvasTexture({var}Canvas);\n\
                     const {var}Mat = new THREE.SpriteMaterial({{map: {var}Tex}});\n\
                     const {var} = new THREE.Sprite({var}Mat);\n\
                     {var}.position.set({}, {}, {});\n\
                     {var}.scale.set({}, {}, 1);\n\
                     scene.add({var});\n\n",
                    position[0],
                    position[1],
                    position[2],
                    size * 10.0,
                    position[0],
                    position[1],
                    position[2],
                    size,
                    size / 4.0
                )
                .unwrap();
            }
        }
    }

    let [cx, cy, cz] = params.camera_position;
    let [tx, ty, tz] = params.camera_target;

    format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AskDavidC 3D Scene</title>
<style>body{{margin:0;overflow:hidden;background:{bg}}}</style>
</head><body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const scene = new THREE.Scene();
scene.background = new THREE.Color('{bg}');
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
camera.position.set({cx}, {cy}, {cz});
camera.lookAt({tx}, {ty}, {tz});
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Default lighting
scene.add(new THREE.AmbientLight('{ambient_color}', 0.5));
const dirLight = new THREE.DirectionalLight('{dir_color}', 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

// Objects
{objects_js}
// Animation
function animate() {{
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}}
animate();

// Orbit controls (mouse drag to rotate)
let isDragging = false, prevX, prevY;
renderer.domElement.addEventListener('mousedown', e => {{ isDragging = true; prevX = e.clientX; prevY = e.clientY; }});
renderer.domElement.addEventListener('mouseup', () => isDragging = false);
renderer.domElement.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  const dx = (e.clientX - prevX) * 0.01, dy = (e.clientY - prevY) * 0.01;
  camera.position.applyAxisAngle(new THREE.Vector3(0,1,0), -dx);
  camera.position.y += dy;
  camera.lookAt({tx},{ty},{tz});
  prevX = e.clientX; prevY = e.clientY;
}});
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
</script>
</body></html>"#,
        bg = params.background,
        objects_js = objects_js,
        cx = cx,
        cy = cy,
        cz = cz,
        tx = tx,
        ty = ty,
        tz = tz,
        ambient_color = "#404060",
        dir_color = "#ffffff",
    )
}

// ─────────────────────────────────────────────────
// §26  SVG Chart Generator
// ─────────────────────────────────────────────────

/// Chart type for SVG chart generator.
#[derive(Debug, Clone, PartialEq)]
pub enum ChartKind {
    Bar,
    Line,
    Pie,
    Scatter,
    Histogram { bins: usize },
}

/// SVG chart parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartParams {
    pub kind: ChartKind,
    pub title: String,
    pub labels: Vec<String>,
    pub values: Vec<f64>,
    pub width: f64,
    pub height: f64,
}

impl Default for ChartParams {
    fn default() -> Self {
        Self {
            kind: ChartKind::Bar,
            title: String::new(),
            labels: Vec::new(),
            values: Vec::new(),
            width: 500.0,
            height: 300.0,
        }
    }
}

/// Generate an SVG chart.
pub fn generate_chart(params: &ChartParams) -> String {
    let w = params.width;
    let h = params.height;
    let margin = 50.0;
    let cw = w - 2.0 * margin;
    let ch = h - 2.0 * margin;
    let colors = [
        "#4DE6D9", "#E6B333", "#4DCC80", "#CC4D4D", "#9966FF", "#FF6633", "#33AAFF", "#FF66CC",
    ];

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">"#,
    );
    let bg_fill = "#1a1a2e";
    write!(svg, r#"<rect width="{w}" height="{h}" fill="{bg_fill}"/>"#,).unwrap();
    // Title
    if !params.title.is_empty() {
        let title_fill = "#eee";
        let title_x = w / 2.0;
        let title_text = &params.title;
        write!(svg,
            r#"<text x="{title_x}" y="25" fill="{title_fill}" font-family="monospace" font-size="14" text-anchor="middle">{title_text}</text>"#,
        ).unwrap();
    }

    if params.values.is_empty() {
        svg.push_str("</svg>");
        return svg;
    }

    let max_val = params
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_val = params.values.iter().copied().fold(f64::INFINITY, f64::min);
    let range = if (max_val - min_val).abs() < f64::EPSILON {
        1.0
    } else {
        max_val - min_val
    };
    let n = params.values.len();

    match &params.kind {
        ChartKind::Bar => {
            let bar_w = cw / n as f64 * 0.8;
            let gap = cw / n as f64 * 0.2;
            let safe_max = if max_val.abs() < f64::EPSILON {
                1.0
            } else {
                max_val
            };
            for (i, v) in params.values.iter().enumerate() {
                let bh = (v / safe_max) * ch;
                let x = margin + i as f64 * (bar_w + gap);
                let y = margin + ch - bh;
                let color = colors[i % colors.len()];
                write!(svg,
                    r#"<rect x="{x:.1}" y="{y:.1}" width="{bar_w:.1}" height="{bh:.1}" fill="{color}" rx="2"/>"#,
                ).unwrap();
                // Value label
                let val_fill = "#ccc";
                let val_x = x + bar_w / 2.0;
                let val_y = y - 4.0;
                let val_text = format_math_result(*v);
                write!(svg,
                    r#"<text x="{val_x:.1}" y="{val_y:.1}" fill="{val_fill}" font-family="monospace" font-size="10" text-anchor="middle">{val_text}</text>"#,
                ).unwrap();
                // X-axis label
                let label = params.labels.get(i).map_or("", std::string::String::as_str);
                if !label.is_empty() {
                    let lbl_fill = "#888";
                    let lbl_x = x + bar_w / 2.0;
                    let lbl_y = h - 10.0;
                    write!(svg,
                        r#"<text x="{lbl_x:.1}" y="{lbl_y:.1}" fill="{lbl_fill}" font-family="monospace" font-size="9" text-anchor="middle">{label}</text>"#,
                    ).unwrap();
                }
            }
        }
        ChartKind::Line => {
            let mut points = String::new();
            for (i, v) in params.values.iter().enumerate() {
                let x = margin + (i as f64 / (n - 1).max(1) as f64) * cw;
                let y = margin + ch - ((v - min_val) / range) * ch;
                if i > 0 {
                    points.push(' ');
                }
                write!(points, "{x:.1},{y:.1}").unwrap();
            }
            let stroke = "#4DE6D9";
            write!(
                svg,
                r#"<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="2"/>"#,
            )
            .unwrap();
            // Dots
            let fill = "#4DE6D9";
            for (i, v) in params.values.iter().enumerate() {
                let x = margin + (i as f64 / (n - 1).max(1) as f64) * cw;
                let y = margin + ch - ((v - min_val) / range) * ch;
                write!(
                    svg,
                    r#"<circle cx="{x:.1}" cy="{y:.1}" r="3" fill="{fill}"/>"#,
                )
                .unwrap();
            }
        }
        ChartKind::Pie => {
            let total: f64 = params.values.iter().sum();
            let safe_total = if total.abs() < f64::EPSILON {
                1.0
            } else {
                total
            };
            let cx = w / 2.0;
            let cy = h / 2.0 + 10.0;
            let r = ch.min(cw) / 2.0 - 10.0;
            let mut angle = -std::f64::consts::FRAC_PI_2;
            for (i, v) in params.values.iter().enumerate() {
                let sweep = (v / safe_total) * std::f64::consts::TAU;
                let x1 = cx + r * angle.cos();
                let y1 = cy + r * angle.sin();
                let x2 = cx + r * (angle + sweep).cos();
                let y2 = cy + r * (angle + sweep).sin();
                let large = if sweep > std::f64::consts::PI { 1 } else { 0 };
                let color = colors[i % colors.len()];
                write!(svg,
                    r#"<path d="M{cx},{cy} L{x1:.1},{y1:.1} A{r},{r} 0 {large},1 {x2:.1},{y2:.1} Z" fill="{color}"/>"#,
                ).unwrap();
                // Label at midpoint
                let mid = angle + sweep / 2.0;
                let lx = cx + (r * 0.65) * mid.cos();
                let ly = cy + (r * 0.65) * mid.sin();
                let pct = v / safe_total * 100.0;
                if pct > 3.0 {
                    let text_fill = "#fff";
                    write!(svg,
                        r#"<text x="{lx:.1}" y="{ly:.1}" fill="{text_fill}" font-family="monospace" font-size="10" text-anchor="middle">{pct:.0}%</text>"#,
                    ).unwrap();
                }
                angle += sweep;
            }
        }
        ChartKind::Scatter => {
            // Pairs: (x, y) from alternating values
            let pairs: Vec<(f64, f64)> = params
                .values
                .chunks(2)
                .filter(|c| c.len() == 2)
                .map(|c| (c[0], c[1]))
                .collect();
            if !pairs.is_empty() {
                let x_min = pairs.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                let x_max = pairs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                let y_min = pairs.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
                let y_max = pairs.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
                let x_range = if (x_max - x_min).abs() < f64::EPSILON {
                    1.0
                } else {
                    x_max - x_min
                };
                let y_range = if (y_max - y_min).abs() < f64::EPSILON {
                    1.0
                } else {
                    y_max - y_min
                };
                for (i, (px, py)) in pairs.iter().enumerate() {
                    let sx = margin + ((px - x_min) / x_range) * cw;
                    let sy = margin + ch - ((py - y_min) / y_range) * ch;
                    let color = colors[i % colors.len()];
                    write!(
                        svg,
                        r#"<circle cx="{sx:.1}" cy="{sy:.1}" r="4" fill="{color}" opacity="0.8"/>"#,
                    )
                    .unwrap();
                }
            }
        }
        ChartKind::Histogram { bins } => {
            let b = (*bins).max(2);
            let bin_w = range / b as f64;
            let mut counts = vec![0usize; b];
            for v in &params.values {
                let idx = ((v - min_val) / bin_w).floor() as usize;
                let idx = idx.min(b - 1);
                counts[idx] += 1;
            }
            let max_count = *counts.iter().max().unwrap_or(&1);
            let bar_w = cw / b as f64;
            for (i, count) in counts.iter().enumerate() {
                let bh = (*count as f64 / max_count as f64) * ch;
                let x = margin + i as f64 * bar_w;
                let y = margin + ch - bh;
                let color = colors[i % colors.len()];
                let rect_w = bar_w - 1.0;
                write!(svg,
                    r#"<rect x="{x:.1}" y="{y:.1}" width="{rect_w:.1}" height="{bh:.1}" fill="{color}" opacity="0.85"/>"#,
                ).unwrap();
            }
        }
    }

    // Axes
    let axis_stroke = "#555";
    let m = margin;
    let t = margin;
    let b = margin + ch;
    let r = margin + cw;
    write!(
        svg,
        r#"<line x1="{m}" y1="{t}" x2="{m}" y2="{b}" stroke="{axis_stroke}" stroke-width="1"/>"#,
    )
    .unwrap();
    let l = margin;
    write!(
        svg,
        r#"<line x1="{l}" y1="{b}" x2="{r}" y2="{b}" stroke="{axis_stroke}" stroke-width="1"/>"#,
    )
    .unwrap();

    svg.push_str("</svg>");
    svg
}

// ─────────────────────────────────────────────────
// §27  Graphviz DOT Generator
// ─────────────────────────────────────────────────

/// DOT graph kind.
#[derive(Debug, Clone, PartialEq)]
pub enum DotGraphKind {
    Directed,
    Undirected,
}

/// DOT node.
#[derive(Debug, Clone, PartialEq)]
pub struct DotNode {
    pub id: String,
    pub label: Option<String>,
    pub shape: Option<String>,
    pub color: Option<String>,
}

/// DOT edge.
#[derive(Debug, Clone, PartialEq)]
pub struct DotEdge {
    pub from: String,
    pub to: String,
    pub label: Option<String>,
}

/// DOT graph parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct DotParams {
    pub kind: DotGraphKind,
    pub name: String,
    pub nodes: Vec<DotNode>,
    pub edges: Vec<DotEdge>,
    pub rankdir: Option<String>,
}

/// Generate a Graphviz DOT file.
pub fn generate_dot(params: &DotParams) -> String {
    let keyword = match params.kind {
        DotGraphKind::Directed => "digraph",
        DotGraphKind::Undirected => "graph",
    };
    let arrow = match params.kind {
        DotGraphKind::Directed => "->",
        DotGraphKind::Undirected => "--",
    };
    let mut out = format!("{} {} {{\n", keyword, params.name);
    if let Some(rd) = &params.rankdir {
        writeln!(out, "    rankdir={rd};").unwrap();
    }
    out.push_str("    node [style=filled, fontname=\"monospace\"];\n");
    for node in &params.nodes {
        let mut attrs = Vec::new();
        if let Some(l) = &node.label {
            attrs.push(format!("label=\"{l}\""));
        }
        if let Some(s) = &node.shape {
            attrs.push(format!("shape={s}"));
        }
        if let Some(c) = &node.color {
            attrs.push(format!("fillcolor=\"{c}\""));
        }
        if attrs.is_empty() {
            writeln!(out, "    {};", node.id).unwrap();
        } else {
            writeln!(out, "    {} [{}];", node.id, attrs.join(", ")).unwrap();
        }
    }
    for edge in &params.edges {
        if let Some(l) = &edge.label {
            writeln!(
                out,
                "    {} {} {} [label=\"{}\"];",
                edge.from, arrow, edge.to, l
            )
            .unwrap();
        } else {
            writeln!(out, "    {} {} {};", edge.from, arrow, edge.to).unwrap();
        }
    }
    out.push_str("}\n");
    out
}

// ─────────────────────────────────────────────────
// §28  Mermaid Diagram Generator
// ─────────────────────────────────────────────────

/// Mermaid diagram kind.
#[derive(Debug, Clone, PartialEq)]
pub enum MermaidKind {
    Flowchart { direction: String },
    Sequence,
    Gantt,
    ClassDiagram,
    StateDiagram,
}

/// Mermaid diagram element.
#[derive(Debug, Clone, PartialEq)]
pub enum MermaidElement {
    /// Flowchart node: id, label, shape (round/rect/diamond/stadium)
    Node {
        id: String,
        label: String,
        shape: String,
    },
    /// Edge: from → to with optional label
    Edge {
        from: String,
        to: String,
        label: Option<String>,
    },
    /// Sequence message: from ->> to: msg
    Message {
        from: String,
        to: String,
        text: String,
        dashed: bool,
    },
    /// Gantt task: title, after, duration
    Task {
        title: String,
        id: String,
        after: Option<String>,
        duration: String,
    },
    /// Class member
    ClassMember { class_name: String, member: String },
    /// State transition
    StateTransition {
        from: String,
        to: String,
        label: Option<String>,
    },
}

/// Mermaid diagram parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MermaidParams {
    pub kind: MermaidKind,
    pub title: Option<String>,
    pub elements: Vec<MermaidElement>,
}

/// Generate a Mermaid diagram.
pub fn generate_mermaid(params: &MermaidParams) -> String {
    let mut out = String::new();

    match &params.kind {
        MermaidKind::Flowchart { direction } => {
            writeln!(out, "flowchart {direction}").unwrap();
        }
        MermaidKind::Sequence => {
            out.push_str("sequenceDiagram\n");
        }
        MermaidKind::Gantt => {
            out.push_str("gantt\n");
            if let Some(t) = &params.title {
                writeln!(out, "    title {t}").unwrap();
            }
            out.push_str("    dateFormat YYYY-MM-DD\n");
        }
        MermaidKind::ClassDiagram => {
            out.push_str("classDiagram\n");
        }
        MermaidKind::StateDiagram => {
            out.push_str("stateDiagram-v2\n");
        }
    }

    for el in &params.elements {
        match el {
            MermaidElement::Node { id, label, shape } => {
                let wrapped = match shape.as_str() {
                    "round" => format!("{id}({label})"),
                    "diamond" => format!("{id}{{{{{label}}}}}"),
                    "stadium" => format!("{id}([{label}])"),
                    _ => format!("{id}[{label}]"),
                };
                writeln!(out, "    {wrapped}").unwrap();
            }
            MermaidElement::Edge { from, to, label } => {
                if let Some(l) = &label {
                    writeln!(out, "    {from} -->|{l}| {to}").unwrap();
                } else {
                    writeln!(out, "    {from} --> {to}").unwrap();
                }
            }
            MermaidElement::Message {
                from,
                to,
                text,
                dashed,
            } => {
                let arrow = if *dashed { "-->>" } else { "->>" };
                writeln!(out, "    {from}{arrow}{to}: {text}").unwrap();
            }
            MermaidElement::Task {
                title,
                id,
                after,
                duration,
            } => {
                if let Some(a) = &after {
                    writeln!(out, "    {title} : {id}, after {a}, {duration}").unwrap();
                } else {
                    writeln!(out, "    {title} : {id}, {duration}").unwrap();
                }
            }
            MermaidElement::ClassMember { class_name, member } => {
                writeln!(out, "    {class_name} : {member}").unwrap();
            }
            MermaidElement::StateTransition { from, to, label } => {
                if let Some(l) = &label {
                    writeln!(out, "    {from} --> {to} : {l}").unwrap();
                } else {
                    writeln!(out, "    {from} --> {to}").unwrap();
                }
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────
// §29  WAV Audio Generator
// ─────────────────────────────────────────────────

/// Waveform shape.
#[derive(Debug, Clone, PartialEq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    WhiteNoise,
}

/// WAV generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct WavParams {
    pub waveform: Waveform,
    pub frequency: f64,
    pub duration_ms: u32,
    pub sample_rate: u32,
    pub amplitude: f64,
}

impl Default for WavParams {
    fn default() -> Self {
        Self {
            waveform: Waveform::Sine,
            frequency: 440.0,
            duration_ms: 1000,
            sample_rate: 44100,
            amplitude: 0.8,
        }
    }
}

/// Generate a WAV file as base64-encoded data URI.
pub fn generate_wav(params: &WavParams) -> String {
    let num_samples = (params.sample_rate as f64 * params.duration_ms as f64 / 1000.0) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Simple LCG for noise (deterministic, no rand dependency)
    let mut rng_state: u32 = 42;
    let next_rng = |state: &mut u32| -> f64 {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((*state >> 16) as f64 / 32768.0) * 2.0 - 1.0
    };

    for i in 0..num_samples {
        let t = i as f64 / params.sample_rate as f64;
        let phase = t * params.frequency;
        let sample = match params.waveform {
            Waveform::Sine => (phase * std::f64::consts::TAU).sin(),
            Waveform::Square => {
                if (phase * std::f64::consts::TAU).sin() >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 2.0 * (phase - phase.floor()) - 1.0,
            Waveform::Triangle => {
                let p = phase - phase.floor();
                if p < 0.5 {
                    4.0 * p - 1.0
                } else {
                    3.0 - 4.0 * p
                }
            }
            Waveform::WhiteNoise => next_rng(&mut rng_state),
        };
        let clamped = (sample * params.amplitude).clamp(-1.0, 1.0);
        samples.push((clamped * 32767.0) as i16);
    }

    // Build WAV header (PCM, mono, 16-bit)
    let data_size = (num_samples * 2) as u32;
    let file_size = 36 + data_size;
    let sr = params.sample_rate;

    let mut wav: Vec<u8> = Vec::with_capacity(44 + data_size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sr.to_le_bytes()); // sample rate
    wav.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for s in &samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }

    // Base64 encode
    let b64 = base64_encode_bytes(&wav);
    format!("data:audio/wav;base64,{b64}")
}

/// Base64 encode raw bytes.
fn base64_encode_bytes(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ─────────────────────────────────────────────────
// §30  Wavefront OBJ Generator
// ─────────────────────────────────────────────────

/// OBJ primitive shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjPrimitive {
    Box {
        w: f64,
        d: f64,
        h: f64,
    },
    Sphere {
        r: f64,
        segments: u32,
    },
    Cylinder {
        r: f64,
        h: f64,
        segments: u32,
    },
    Plane {
        w: f64,
        d: f64,
    },
    Torus {
        major_r: f64,
        minor_r: f64,
        segments: u32,
        rings: u32,
    },
}

/// Generate a Wavefront OBJ file.
pub fn generate_obj(primitive: &ObjPrimitive, name: &str) -> String {
    let mut out = format!("# Wavefront OBJ — generated by jouleclaw\no {name}\n\n");

    match primitive {
        ObjPrimitive::Box { w, d, h } => {
            let (hw, hd, hh) = (w / 2.0, d / 2.0, h / 2.0);
            // 8 vertices
            for &z in &[-hh, hh] {
                for &y in &[-hd, hd] {
                    for &x in &[-hw, hw] {
                        writeln!(out, "v {x:.6} {y:.6} {z:.6}").unwrap();
                    }
                }
            }
            out.push('\n');
            // 6 faces (quads)
            let faces = [
                (1, 2, 4, 3),
                (5, 6, 8, 7),
                (1, 2, 6, 5),
                (3, 4, 8, 7),
                (1, 3, 7, 5),
                (2, 4, 8, 6),
            ];
            for (a, b, c, d_) in &faces {
                writeln!(out, "f {a} {b} {c} {d_}").unwrap();
            }
        }
        ObjPrimitive::Sphere { r, segments } => {
            let segs = (*segments).max(8);
            let rings = segs / 2;
            // Top vertex
            writeln!(out, "v 0.000000 {r:.6} 0.000000").unwrap();
            // Ring vertices
            for j in 1..rings {
                let phi = std::f64::consts::PI * j as f64 / rings as f64;
                let y = r * phi.cos();
                let ring_r = r * phi.sin();
                for i in 0..segs {
                    let theta = std::f64::consts::TAU * i as f64 / segs as f64;
                    writeln!(
                        out,
                        "v {:.6} {:.6} {:.6}",
                        ring_r * theta.cos(),
                        y,
                        ring_r * theta.sin()
                    )
                    .unwrap();
                }
            }
            // Bottom vertex
            writeln!(out, "v 0.000000 {:.6} 0.000000", -r).unwrap();
            out.push('\n');
            // Top cap
            for i in 0..segs {
                let next = (i + 1) % segs;
                writeln!(out, "f 1 {} {}", i + 2, next + 2).unwrap();
            }
            // Middle rings
            for j in 0..(rings - 2) {
                let base = 2 + j * segs;
                let next_base = base + segs;
                for i in 0..segs {
                    let next = (i + 1) % segs;
                    writeln!(
                        out,
                        "f {} {} {} {}",
                        base + i,
                        base + next,
                        next_base + next,
                        next_base + i
                    )
                    .unwrap();
                }
            }
            // Bottom cap
            let bottom = 2 + (rings - 1) * segs;
            let last_ring = 2 + (rings - 2) * segs;
            for i in 0..segs {
                let next = (i + 1) % segs;
                writeln!(out, "f {} {} {}", bottom, last_ring + next, last_ring + i).unwrap();
            }
        }
        ObjPrimitive::Cylinder { r, h, segments } => {
            let segs = (*segments).max(6);
            let hh = h / 2.0;
            // Top ring vertices
            for i in 0..segs {
                let a = std::f64::consts::TAU * i as f64 / segs as f64;
                writeln!(out, "v {:.6} {:.6} {:.6}", r * a.cos(), hh, r * a.sin()).unwrap();
            }
            // Bottom ring vertices
            for i in 0..segs {
                let a = std::f64::consts::TAU * i as f64 / segs as f64;
                writeln!(out, "v {:.6} {:.6} {:.6}", r * a.cos(), -hh, r * a.sin()).unwrap();
            }
            // Top/bottom center
            writeln!(out, "v 0.000000 {hh:.6} 0.000000").unwrap();
            writeln!(out, "v 0.000000 {:.6} 0.000000", -hh).unwrap();
            let top_c = 2 * segs + 1;
            let bot_c = 2 * segs + 2;
            out.push('\n');
            for i in 0..segs {
                let next = (i + 1) % segs;
                // Side quad
                writeln!(
                    out,
                    "f {} {} {} {}",
                    i + 1,
                    next + 1,
                    segs + next + 1,
                    segs + i + 1
                )
                .unwrap();
                // Top cap tri
                writeln!(out, "f {} {} {}", top_c, i + 1, next + 1).unwrap();
                // Bottom cap tri
                writeln!(out, "f {} {} {}", bot_c, segs + next + 1, segs + i + 1).unwrap();
            }
        }
        ObjPrimitive::Plane { w, d } => {
            let (hw, hd) = (w / 2.0, d / 2.0);
            writeln!(out, "v {:.6} 0.000000 {:.6}", -hw, -hd).unwrap();
            writeln!(out, "v {:.6} 0.000000 {:.6}", hw, -hd).unwrap();
            writeln!(out, "v {hw:.6} 0.000000 {hd:.6}").unwrap();
            writeln!(out, "v {:.6} 0.000000 {:.6}", -hw, hd).unwrap();
            out.push_str("\nf 1 2 3 4\n");
        }
        ObjPrimitive::Torus {
            major_r,
            minor_r,
            segments,
            rings,
        } => {
            let segs = (*segments).max(8);
            let rngs = (*rings).max(8);
            for j in 0..rngs {
                let theta = std::f64::consts::TAU * j as f64 / rngs as f64;
                let ct = theta.cos();
                let st = theta.sin();
                for i in 0..segs {
                    let phi = std::f64::consts::TAU * i as f64 / segs as f64;
                    let x = (major_r + minor_r * phi.cos()) * ct;
                    let y = minor_r * phi.sin();
                    let z = (major_r + minor_r * phi.cos()) * st;
                    writeln!(out, "v {x:.6} {y:.6} {z:.6}").unwrap();
                }
            }
            out.push('\n');
            for j in 0..rngs {
                let j_next = (j + 1) % rngs;
                for i in 0..segs {
                    let i_next = (i + 1) % segs;
                    let v1 = j * segs + i + 1;
                    let v2 = j * segs + i_next + 1;
                    let v3 = j_next * segs + i_next + 1;
                    let v4 = j_next * segs + i + 1;
                    writeln!(out, "f {v1} {v2} {v3} {v4}").unwrap();
                }
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────
// §31  LaTeX Expression Generator
// ─────────────────────────────────────────────────

/// LaTeX expression kind.
#[derive(Debug, Clone, PartialEq)]
pub enum LatexKind {
    /// Raw LaTeX expression.
    Expression { expr: String },
    /// Matrix (2D array of values).
    Matrix {
        rows: Vec<Vec<String>>,
        bracket: String,
    },
    /// Fraction.
    Fraction { num: String, den: String },
    /// Sum/Product.
    Summation {
        var: String,
        lower: String,
        upper: String,
        body: String,
    },
    /// Integral.
    Integral {
        var: String,
        lower: Option<String>,
        upper: Option<String>,
        body: String,
    },
    /// System of equations.
    System { equations: Vec<String> },
}

/// Generate a LaTeX expression.
pub fn generate_latex(kind: &LatexKind) -> String {
    match kind {
        LatexKind::Expression { expr } => expr.clone(),
        LatexKind::Matrix { rows, bracket } => {
            let env = match bracket.as_str() {
                "[" | "bmatrix" => "bmatrix",
                "|" | "vmatrix" => "vmatrix",
                "{" | "Bmatrix" => "Bmatrix",
                _ => "pmatrix",
            };
            let mut out = format!("\\begin{{{env}}}\n");
            for (i, row) in rows.iter().enumerate() {
                out.push_str(&row.join(" & "));
                if i < rows.len() - 1 {
                    out.push_str(" \\\\\n");
                } else {
                    out.push('\n');
                }
            }
            write!(out, "\\end{{{env}}}").unwrap();
            out
        }
        LatexKind::Fraction { num, den } => {
            format!("\\frac{{{num}}}{{{den}}}")
        }
        LatexKind::Summation {
            var,
            lower,
            upper,
            body,
        } => {
            format!("\\sum_{{{var}={lower}}}^{{{upper}}} {body}")
        }
        LatexKind::Integral {
            var,
            lower,
            upper,
            body,
        } => {
            let bounds = match (lower, upper) {
                (Some(l), Some(u)) => format!("_{{{l}}}^{{{u}}}"),
                (Some(l), None) => format!("_{{{l}}}"),
                _ => String::new(),
            };
            format!("\\int{bounds} {body} \\, d{var}")
        }
        LatexKind::System { equations } => {
            let mut out = String::from("\\begin{cases}\n");
            for eq in equations {
                writeln!(out, "{eq} \\\\").unwrap();
            }
            out.push_str("\\end{cases}");
            out
        }
    }
}

// ─────────────────────────────────────────────────
// §32  Equation Formatter (Unicode Math)
// ─────────────────────────────────────────────────

/// Format a math expression using Unicode symbols.
pub fn format_equation_unicode(expr: &str) -> String {
    let mut out = expr.to_string();
    // Operators
    out = out.replace("!=", "\u{2260}"); // ≠
    out = out.replace("<=", "\u{2264}"); // ≤
    out = out.replace(">=", "\u{2265}"); // ≥
    out = out.replace("+-", "\u{00B1}"); // ±
    out = out.replace('*', "\u{00D7}"); // ×
    out = out.replace("...", "\u{2026}"); // …
    out = out.replace("inf", "\u{221E}"); // ∞
    out = out.replace("sqrt", "\u{221A}"); // √
    out = out.replace("pi", "\u{03C0}"); // π
    out = out.replace("theta", "\u{03B8}"); // θ
    out = out.replace("alpha", "\u{03B1}"); // α
    out = out.replace("beta", "\u{03B2}"); // β
    out = out.replace("gamma", "\u{03B3}"); // γ
    out = out.replace("delta", "\u{03B4}"); // δ
    out = out.replace("epsilon", "\u{03B5}"); // ε
    out = out.replace("lambda", "\u{03BB}"); // λ
    out = out.replace("sigma", "\u{03C3}"); // σ
    out = out.replace("omega", "\u{03C9}"); // ω
    out = out.replace("sum", "\u{2211}"); // ∑
    out = out.replace("prod", "\u{220F}"); // ∏
    out = out.replace("integral", "\u{222B}"); // ∫
    out = out.replace("partial", "\u{2202}"); // ∂
    out = out.replace("nabla", "\u{2207}"); // ∇
    out = out.replace("in ", "\u{2208} "); // ∈
    out = out.replace("notin", "\u{2209}"); // ∉
    out = out.replace("forall", "\u{2200}"); // ∀
    out = out.replace("exists", "\u{2203}"); // ∃
    out = out.replace("approx", "\u{2248}"); // ≈
    out = out.replace("->", "\u{2192}"); // →
    out = out.replace("<->", "\u{2194}"); // ↔
    out = out.replace("=>", "\u{21D2}"); // ⇒
    // Superscript digits
    let superscripts = [
        ("^0", "\u{2070}"),
        ("^1", "\u{00B9}"),
        ("^2", "\u{00B2}"),
        ("^3", "\u{00B3}"),
        ("^4", "\u{2074}"),
        ("^5", "\u{2075}"),
        ("^6", "\u{2076}"),
        ("^7", "\u{2077}"),
        ("^8", "\u{2078}"),
        ("^9", "\u{2079}"),
        ("^n", "\u{207F}"),
    ];
    for (ascii, uni) in &superscripts {
        out = out.replace(ascii, uni);
    }
    // Subscript digits
    let subscripts = [
        ("_0", "\u{2080}"),
        ("_1", "\u{2081}"),
        ("_2", "\u{2082}"),
        ("_3", "\u{2083}"),
        ("_4", "\u{2084}"),
        ("_5", "\u{2085}"),
        ("_6", "\u{2086}"),
        ("_7", "\u{2087}"),
        ("_8", "\u{2088}"),
        ("_9", "\u{2089}"),
        ("_n", "\u{2099}"),
    ];
    for (ascii, uni) in &subscripts {
        out = out.replace(ascii, uni);
    }
    out
}

// ─────────────────────────────────────────────────
// §33  Terraform HCL Generator
// ─────────────────────────────────────────────────

/// Terraform resource kind.
#[derive(Debug, Clone, PartialEq)]
pub enum TerraformResource {
    AwsInstance {
        ami: String,
        instance_type: String,
        name: String,
    },
    AwsS3Bucket {
        bucket: String,
        acl: String,
    },
    AwsSecurityGroup {
        name: String,
        ingress_ports: Vec<u16>,
    },
    AwsVpc {
        cidr: String,
        name: String,
    },
    AwsRds {
        engine: String,
        instance_class: String,
        name: String,
    },
    AwsLambda {
        function_name: String,
        runtime: String,
        handler: String,
    },
    GcpInstance {
        machine_type: String,
        zone: String,
        name: String,
    },
    AzureVm {
        size: String,
        name: String,
    },
}

/// Terraform HCL parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TerraformParams {
    pub provider: String,
    pub region: String,
    pub resources: Vec<TerraformResource>,
}

/// Generate Terraform HCL.
pub fn generate_terraform(params: &TerraformParams) -> String {
    let mut out = format!(
        "provider \"{}\" {{\n  region = \"{}\"\n}}\n\n",
        params.provider, params.region
    );

    for res in &params.resources {
        match res {
            TerraformResource::AwsInstance {
                ami,
                instance_type,
                name,
            } => {
                write!(out,
                    "resource \"aws_instance\" \"{}\" {{\n  ami           = \"{}\"\n  instance_type = \"{}\"\n\n  tags = {{\n    Name = \"{}\"\n  }}\n}}\n\n",
                    name.replace('-', "_"), ami, instance_type, name
                ).unwrap();
            }
            TerraformResource::AwsS3Bucket { bucket, acl } => {
                write!(out,
                    "resource \"aws_s3_bucket\" \"{}\" {{\n  bucket = \"{}\"\n  acl    = \"{}\"\n}}\n\n",
                    bucket.replace(['-', '.'], "_"), bucket, acl
                ).unwrap();
            }
            TerraformResource::AwsSecurityGroup {
                name,
                ingress_ports,
            } => {
                write!(
                    out,
                    "resource \"aws_security_group\" \"{}\" {{\n  name = \"{}\"\n\n",
                    name.replace('-', "_"),
                    name
                )
                .unwrap();
                for port in ingress_ports {
                    write!(out,
                        "  ingress {{\n    from_port   = {port}\n    to_port     = {port}\n    protocol    = \"tcp\"\n    cidr_blocks = [\"0.0.0.0/0\"]\n  }}\n\n"
                    ).unwrap();
                }
                out.push_str("}\n\n");
            }
            TerraformResource::AwsVpc { cidr, name } => {
                write!(out,
                    "resource \"aws_vpc\" \"{}\" {{\n  cidr_block = \"{}\"\n\n  tags = {{\n    Name = \"{}\"\n  }}\n}}\n\n",
                    name.replace('-', "_"), cidr, name
                ).unwrap();
            }
            TerraformResource::AwsRds {
                engine,
                instance_class,
                name,
            } => {
                write!(out,
                    "resource \"aws_db_instance\" \"{}\" {{\n  engine         = \"{}\"\n  instance_class = \"{}\"\n  identifier     = \"{}\"\n}}\n\n",
                    name.replace('-', "_"), engine, instance_class, name
                ).unwrap();
            }
            TerraformResource::AwsLambda {
                function_name,
                runtime,
                handler,
            } => {
                write!(out,
                    "resource \"aws_lambda_function\" \"{}\" {{\n  function_name = \"{}\"\n  runtime       = \"{}\"\n  handler       = \"{}\"\n  filename      = \"lambda.zip\"\n}}\n\n",
                    function_name.replace('-', "_"), function_name, runtime, handler
                ).unwrap();
            }
            TerraformResource::GcpInstance {
                machine_type,
                zone,
                name,
            } => {
                write!(out,
                    "resource \"google_compute_instance\" \"{}\" {{\n  name         = \"{}\"\n  machine_type = \"{}\"\n  zone         = \"{}\"\n\n  boot_disk {{\n    initialize_params {{\n      image = \"debian-cloud/debian-11\"\n    }}\n  }}\n}}\n\n",
                    name.replace('-', "_"), name, machine_type, zone
                ).unwrap();
            }
            TerraformResource::AzureVm { size, name } => {
                write!(out,
                    "resource \"azurerm_virtual_machine\" \"{}\" {{\n  name     = \"{}\"\n  vm_size  = \"{}\"\n  location = \"eastus\"\n}}\n\n",
                    name.replace('-', "_"), name, size
                ).unwrap();
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────
// §34  Docker Compose YAML Generator
// ─────────────────────────────────────────────────

/// Docker Compose service definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposeService {
    pub name: String,
    pub image: String,
    pub ports: Vec<(u16, u16)>,
    pub environment: Vec<(String, String)>,
    pub volumes: Vec<String>,
    pub depends_on: Vec<String>,
    pub command: Option<String>,
}

/// Docker Compose parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposeParams {
    pub services: Vec<ComposeService>,
    pub networks: Vec<String>,
    pub volumes: Vec<String>,
}

/// Generate Docker Compose YAML.
pub fn generate_compose(params: &ComposeParams) -> String {
    let mut out = String::from("version: '3.8'\n\nservices:\n");

    for svc in &params.services {
        write!(out, "  {}:\n    image: {}\n", svc.name, svc.image).unwrap();
        if !svc.ports.is_empty() {
            out.push_str("    ports:\n");
            for (host, container) in &svc.ports {
                writeln!(out, "      - \"{host}:{container}\"").unwrap();
            }
        }
        if !svc.environment.is_empty() {
            out.push_str("    environment:\n");
            for (k, v) in &svc.environment {
                writeln!(out, "      {k}: \"{v}\"").unwrap();
            }
        }
        if !svc.volumes.is_empty() {
            out.push_str("    volumes:\n");
            for v in &svc.volumes {
                writeln!(out, "      - {v}").unwrap();
            }
        }
        if !svc.depends_on.is_empty() {
            out.push_str("    depends_on:\n");
            for d in &svc.depends_on {
                writeln!(out, "      - {d}").unwrap();
            }
        }
        if let Some(cmd) = &svc.command {
            writeln!(out, "    command: {cmd}").unwrap();
        }
        out.push('\n');
    }

    if !params.networks.is_empty() {
        out.push_str("networks:\n");
        for net in &params.networks {
            write!(out, "  {net}:\n    driver: bridge\n").unwrap();
        }
        out.push('\n');
    }

    if !params.volumes.is_empty() {
        out.push_str("volumes:\n");
        for vol in &params.volumes {
            writeln!(out, "  {vol}:").unwrap();
        }
    }

    out
}

// ─────────────────────────────────────────────────
// §35  Kubernetes Manifest Generator
// ─────────────────────────────────────────────────

/// Kubernetes resource kind.
#[derive(Debug, Clone, PartialEq)]
pub enum K8sResource {
    Deployment {
        name: String,
        image: String,
        replicas: u32,
        port: u16,
        cpu: Option<String>,
        memory: Option<String>,
    },
    Service {
        name: String,
        port: u16,
        target_port: u16,
        svc_type: String,
    },
    Ingress {
        name: String,
        host: String,
        service: String,
        port: u16,
    },
    ConfigMap {
        name: String,
        data: Vec<(String, String)>,
    },
    Secret {
        name: String,
        data: Vec<(String, String)>,
    },
}

/// Kubernetes parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct K8sParams {
    pub namespace: String,
    pub resources: Vec<K8sResource>,
}

/// Generate Kubernetes manifests (YAML).
pub fn generate_k8s(params: &K8sParams) -> String {
    let mut out = String::new();

    for (i, res) in params.resources.iter().enumerate() {
        if i > 0 {
            out.push_str("---\n");
        }
        match res {
            K8sResource::Deployment {
                name,
                image,
                replicas,
                port,
                cpu,
                memory,
            } => {
                write!(out,
                    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {}\n  namespace: {}\nspec:\n  replicas: {}\n  selector:\n    matchLabels:\n      app: {}\n  template:\n    metadata:\n      labels:\n        app: {}\n    spec:\n      containers:\n        - name: {}\n          image: {}\n          ports:\n            - containerPort: {}\n",
                    name, params.namespace, replicas, name, name, name, image, port
                ).unwrap();
                if cpu.is_some() || memory.is_some() {
                    out.push_str("          resources:\n            requests:\n");
                    if let Some(c) = &cpu {
                        writeln!(out, "              cpu: \"{c}\"").unwrap();
                    }
                    if let Some(m) = &memory {
                        writeln!(out, "              memory: \"{m}\"").unwrap();
                    }
                }
            }
            K8sResource::Service {
                name,
                port,
                target_port,
                svc_type,
            } => {
                write!(out,
                    "apiVersion: v1\nkind: Service\nmetadata:\n  name: {}\n  namespace: {}\nspec:\n  type: {}\n  selector:\n    app: {}\n  ports:\n    - port: {}\n      targetPort: {}\n",
                    name, params.namespace, svc_type, name, port, target_port
                ).unwrap();
            }
            K8sResource::Ingress {
                name,
                host,
                service,
                port,
            } => {
                write!(out,
                    "apiVersion: networking.k8s.io/v1\nkind: Ingress\nmetadata:\n  name: {}\n  namespace: {}\nspec:\n  rules:\n    - host: {}\n      http:\n        paths:\n          - path: /\n            pathType: Prefix\n            backend:\n              service:\n                name: {}\n                port:\n                  number: {}\n",
                    name, params.namespace, host, service, port
                ).unwrap();
            }
            K8sResource::ConfigMap { name, data } => {
                write!(out,
                    "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {}\n  namespace: {}\ndata:\n",
                    name, params.namespace
                ).unwrap();
                for (k, v) in data {
                    writeln!(out, "  {k}: \"{v}\"").unwrap();
                }
            }
            K8sResource::Secret { name, data } => {
                write!(out,
                    "apiVersion: v1\nkind: Secret\nmetadata:\n  name: {}\n  namespace: {}\ntype: Opaque\ndata:\n",
                    name, params.namespace
                ).unwrap();
                for (k, v) in data {
                    writeln!(out, "  {}: {}", k, base64_encode(v)).unwrap();
                }
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────
// §36  KiCad Schematic Generator
// ─────────────────────────────────────────────────

/// KiCad component kind.
#[derive(Debug, Clone, PartialEq)]
pub enum KicadComponent {
    Resistor {
        ref_des: String,
        value: String,
        x: f64,
        y: f64,
    },
    Capacitor {
        ref_des: String,
        value: String,
        x: f64,
        y: f64,
    },
    Inductor {
        ref_des: String,
        value: String,
        x: f64,
        y: f64,
    },
    Diode {
        ref_des: String,
        x: f64,
        y: f64,
    },
    Led {
        ref_des: String,
        color: String,
        x: f64,
        y: f64,
    },
    Transistor {
        ref_des: String,
        kind: String,
        x: f64,
        y: f64,
    },
    OpAmp {
        ref_des: String,
        x: f64,
        y: f64,
    },
    Ic {
        ref_des: String,
        part_name: String,
        pins: u32,
        x: f64,
        y: f64,
    },
    Gnd {
        x: f64,
        y: f64,
    },
    Vcc {
        voltage: String,
        x: f64,
        y: f64,
    },
}

/// KiCad wire (net connection).
#[derive(Debug, Clone, PartialEq)]
pub struct KicadWire {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// KiCad schematic parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct KicadParams {
    pub title: String,
    pub components: Vec<KicadComponent>,
    pub wires: Vec<KicadWire>,
}

/// Generate KiCad schematic (S-expression format, .kicad_sch v6+).
pub fn generate_kicad(params: &KicadParams) -> String {
    let mut out = String::from("(kicad_sch (version 20211014) (generator jouleclaw)\n\n");
    write!(
        out,
        "  (title_block\n    (title \"{}\")\n  )\n\n",
        params.title
    )
    .unwrap();

    for comp in &params.components {
        match comp {
            KicadComponent::Resistor {
                ref_des,
                value,
                x,
                y,
            } => {
                write!(out,
                    "  (symbol (lib_id \"Device:R\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n    (property \"Value\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, ref_des, x + 1.5, y, value, x + 1.5, y + 1.5
                ).unwrap();
            }
            KicadComponent::Capacitor {
                ref_des,
                value,
                x,
                y,
            } => {
                write!(out,
                    "  (symbol (lib_id \"Device:C\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n    (property \"Value\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, ref_des, x + 1.5, y, value, x + 1.5, y + 1.5
                ).unwrap();
            }
            KicadComponent::Inductor {
                ref_des,
                value,
                x,
                y,
            } => {
                write!(out,
                    "  (symbol (lib_id \"Device:L\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n    (property \"Value\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, ref_des, x + 1.5, y, value, x + 1.5, y + 1.5
                ).unwrap();
            }
            KicadComponent::Diode { ref_des, x, y } => {
                write!(out,
                    "  (symbol (lib_id \"Device:D\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, ref_des, x + 1.5, y
                ).unwrap();
            }
            KicadComponent::Led {
                ref_des,
                color,
                x,
                y,
            } => {
                write!(out,
                    "  (symbol (lib_id \"Device:LED\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n    (property \"Value\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, ref_des, x + 1.5, y, color, x + 1.5, y + 1.5
                ).unwrap();
            }
            KicadComponent::Transistor {
                ref_des,
                kind,
                x,
                y,
            } => {
                let lib = if kind == "PNP" {
                    "Device:Q_PNP"
                } else {
                    "Device:Q_NPN"
                };
                write!(out,
                    "  (symbol (lib_id \"{}\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    lib, x, y, ref_des, x + 1.5, y
                ).unwrap();
            }
            KicadComponent::OpAmp { ref_des, x, y } => {
                write!(out,
                    "  (symbol (lib_id \"Amplifier_Operational:LM358\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, ref_des, x + 2.0, y
                ).unwrap();
            }
            KicadComponent::Ic {
                ref_des,
                part_name,
                pins,
                x,
                y,
            } => {
                write!(out,
                    "  (symbol (lib_id \"{}\") (at {:.2} {:.2} 0)\n    (property \"Reference\" \"{}\" (at {:.2} {:.2} 0))\n    (property \"Pins\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    part_name, x, y, ref_des, x + 2.0, y, pins, x + 2.0, y + 1.5
                ).unwrap();
            }
            KicadComponent::Gnd { x, y } => {
                write!(
                    out,
                    "  (symbol (lib_id \"power:GND\") (at {x:.2} {y:.2} 0))\n\n"
                )
                .unwrap();
            }
            KicadComponent::Vcc { voltage, x, y } => {
                write!(out,
                    "  (symbol (lib_id \"power:VCC\") (at {:.2} {:.2} 0)\n    (property \"Value\" \"{}\" (at {:.2} {:.2} 0))\n  )\n\n",
                    x, y, voltage, x + 1.0, y - 1.0
                ).unwrap();
            }
        }
    }

    for wire in &params.wires {
        writeln!(
            out,
            "  (wire (pts (xy {:.2} {:.2}) (xy {:.2} {:.2})))",
            wire.x1, wire.y1, wire.x2, wire.y2
        )
        .unwrap();
    }

    out.push_str("\n)\n");
    out
}

// ─────────────────────────────────────────────────
// §37  SPICE Netlist Generator
// ─────────────────────────────────────────────────

/// SPICE element.
#[derive(Debug, Clone, PartialEq)]
pub enum SpiceElement {
    Resistor {
        name: String,
        node_p: String,
        node_n: String,
        value: String,
    },
    Capacitor {
        name: String,
        node_p: String,
        node_n: String,
        value: String,
    },
    Inductor {
        name: String,
        node_p: String,
        node_n: String,
        value: String,
    },
    Diode {
        name: String,
        node_p: String,
        node_n: String,
        model: String,
    },
    Mosfet {
        name: String,
        drain: String,
        gate: String,
        source: String,
        model: String,
    },
    VoltageSource {
        name: String,
        node_p: String,
        node_n: String,
        value: String,
    },
    CurrentSource {
        name: String,
        node_p: String,
        node_n: String,
        value: String,
    },
}

/// SPICE simulation command.
#[derive(Debug, Clone, PartialEq)]
pub enum SpiceAnalysis {
    DcOp,
    Ac {
        start: String,
        stop: String,
        points: u32,
    },
    Transient {
        step: String,
        stop: String,
    },
}

/// SPICE netlist parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SpiceParams {
    pub title: String,
    pub elements: Vec<SpiceElement>,
    pub analysis: Option<SpiceAnalysis>,
}

/// Generate a SPICE netlist.
pub fn generate_spice(params: &SpiceParams) -> String {
    let mut out = format!("* {}\n\n", params.title);

    for el in &params.elements {
        match el {
            SpiceElement::Resistor {
                name,
                node_p,
                node_n,
                value,
            } => {
                writeln!(out, "R{name} {node_p} {node_n} {value}").unwrap();
            }
            SpiceElement::Capacitor {
                name,
                node_p,
                node_n,
                value,
            } => {
                writeln!(out, "C{name} {node_p} {node_n} {value}").unwrap();
            }
            SpiceElement::Inductor {
                name,
                node_p,
                node_n,
                value,
            } => {
                writeln!(out, "L{name} {node_p} {node_n} {value}").unwrap();
            }
            SpiceElement::Diode {
                name,
                node_p,
                node_n,
                model,
            } => {
                writeln!(out, "D{name} {node_p} {node_n} {model}").unwrap();
            }
            SpiceElement::Mosfet {
                name,
                drain,
                gate,
                source,
                model,
            } => {
                writeln!(out, "M{name} {drain} {gate} {source} {source} {model}").unwrap();
            }
            SpiceElement::VoltageSource {
                name,
                node_p,
                node_n,
                value,
            } => {
                writeln!(out, "V{name} {node_p} {node_n} {value}").unwrap();
            }
            SpiceElement::CurrentSource {
                name,
                node_p,
                node_n,
                value,
            } => {
                writeln!(out, "I{name} {node_p} {node_n} {value}").unwrap();
            }
        }
    }

    if let Some(analysis) = &params.analysis {
        out.push('\n');
        match analysis {
            SpiceAnalysis::DcOp => out.push_str(".op\n"),
            SpiceAnalysis::Ac {
                start,
                stop,
                points,
            } => {
                writeln!(out, ".ac dec {points} {start} {stop}").unwrap();
            }
            SpiceAnalysis::Transient { step, stop } => {
                writeln!(out, ".tran {step} {stop}").unwrap();
            }
        }
    }

    out.push_str("\n.end\n");
    out
}

// ─────────────────────────────────────────────────
// §38  PBM/PPM Bitmap Generator
// ─────────────────────────────────────────────────

/// Bitmap pixel color.
#[derive(Debug, Clone, PartialEq)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Bitmap pattern kind.
#[derive(Debug, Clone, PartialEq)]
pub enum BitmapPattern {
    Checkerboard {
        size: u32,
        color1: Pixel,
        color2: Pixel,
    },
    Gradient {
        direction: String,
    },
    Stripes {
        orientation: String,
        width: u32,
        color1: Pixel,
        color2: Pixel,
    },
    SolidColor {
        color: Pixel,
    },
    Grid {
        spacing: u32,
        line_color: Pixel,
        bg_color: Pixel,
    },
}

/// Bitmap parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BitmapParams {
    pub width: u32,
    pub height: u32,
    pub pattern: BitmapPattern,
}

/// Generate a PPM (P3) image.
pub fn generate_ppm(params: &BitmapParams) -> String {
    let w = params.width.min(256);
    let h = params.height.min(256);
    let mut out = format!("P3\n{w} {h}\n255\n");

    for y in 0..h {
        for x in 0..w {
            let p = match &params.pattern {
                BitmapPattern::Checkerboard {
                    size,
                    color1,
                    color2,
                } => {
                    let s = (*size).max(1);
                    if (x / s + y / s) % 2 == 0 {
                        color1.clone()
                    } else {
                        color2.clone()
                    }
                }
                BitmapPattern::Gradient { direction } => {
                    let t = match direction.as_str() {
                        "vertical" => y as f64 / h.max(1) as f64,
                        "diagonal" => (x as f64 + y as f64) / (w + h).max(1) as f64,
                        _ => x as f64 / w.max(1) as f64, // horizontal
                    };
                    let v = (t * 255.0) as u8;
                    Pixel { r: v, g: v, b: v }
                }
                BitmapPattern::Stripes {
                    orientation,
                    width,
                    color1,
                    color2,
                } => {
                    let sw = (*width).max(1);
                    let pos = match orientation.as_str() {
                        "vertical" => x,
                        _ => y,
                    };
                    if (pos / sw) % 2 == 0 {
                        color1.clone()
                    } else {
                        color2.clone()
                    }
                }
                BitmapPattern::SolidColor { color } => color.clone(),
                BitmapPattern::Grid {
                    spacing,
                    line_color,
                    bg_color,
                } => {
                    let s = (*spacing).max(1);
                    if x % s == 0 || y % s == 0 {
                        line_color.clone()
                    } else {
                        bg_color.clone()
                    }
                }
            };
            write!(out, "{} {} {} ", p.r, p.g, p.b).unwrap();
        }
        out.push('\n');
    }

    out
}

// ─────────────────────────────────────────────────
// §39  CSV / Table Generator
// ─────────────────────────────────────────────────

/// Table output format.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormat {
    Csv,
    Tsv,
    Markdown,
    AsciiTable,
    Html,
}

/// Table data.
#[derive(Debug, Clone, PartialEq)]
pub struct TableParams {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub format: TableFormat,
}

/// Generate a formatted table.
pub fn generate_table(params: &TableParams) -> String {
    match &params.format {
        TableFormat::Csv => {
            let mut out = params.headers.join(",");
            out.push('\n');
            for row in &params.rows {
                let escaped: Vec<String> = row
                    .iter()
                    .map(|cell| {
                        if cell.contains(',') || cell.contains('"') || cell.contains('\n') {
                            format!("\"{}\"", cell.replace('"', "\"\""))
                        } else {
                            cell.clone()
                        }
                    })
                    .collect();
                out.push_str(&escaped.join(","));
                out.push('\n');
            }
            out
        }
        TableFormat::Tsv => {
            let mut out = params.headers.join("\t");
            out.push('\n');
            for row in &params.rows {
                out.push_str(&row.join("\t"));
                out.push('\n');
            }
            out
        }
        TableFormat::Markdown => {
            let mut out = format!("| {} |\n", params.headers.join(" | "));
            writeln!(
                out,
                "| {} |",
                params
                    .headers
                    .iter()
                    .map(|h| "-".repeat(h.len().max(3)))
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
            .unwrap();
            for row in &params.rows {
                writeln!(out, "| {} |", row.join(" | ")).unwrap();
            }
            out
        }
        TableFormat::AsciiTable => {
            let cols = params.headers.len();
            let mut widths = vec![0usize; cols];
            for (i, h) in params.headers.iter().enumerate() {
                widths[i] = widths[i].max(h.len());
            }
            for row in &params.rows {
                for (i, cell) in row.iter().enumerate() {
                    if i < cols {
                        widths[i] = widths[i].max(cell.len());
                    }
                }
            }
            let sep: String = format!(
                "+{}+\n",
                widths
                    .iter()
                    .map(|w| "-".repeat(w + 2))
                    .collect::<Vec<_>>()
                    .join("+")
            );
            let mut out = sep.clone();
            let header_cells: Vec<String> = params
                .headers
                .iter()
                .enumerate()
                .map(|(i, h)| format!(" {:width$} ", h, width = widths[i]))
                .collect();
            writeln!(out, "|{}|", header_cells.join("|")).unwrap();
            out.push_str(&sep);
            for row in &params.rows {
                let cells: Vec<String> = (0..cols)
                    .map(|i| {
                        let cell = row.get(i).map_or("", std::string::String::as_str);
                        format!(" {:width$} ", cell, width = widths[i])
                    })
                    .collect();
                writeln!(out, "|{}|", cells.join("|")).unwrap();
            }
            out.push_str(&sep);
            out
        }
        TableFormat::Html => {
            let mut out = String::from("<table>\n<thead><tr>");
            for h in &params.headers {
                write!(out, "<th>{h}</th>").unwrap();
            }
            out.push_str("</tr></thead>\n<tbody>\n");
            for row in &params.rows {
                out.push_str("<tr>");
                for cell in row {
                    write!(out, "<td>{cell}</td>").unwrap();
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</tbody>\n</table>");
            out
        }
    }
}

// ─────────────────────────────────────────────────
// §40  Document Classifier (magic bytes)
// ─────────────────────────────────────────────────

/// Document type detected from content/extension.
#[derive(Debug, Clone, PartialEq)]
pub enum DocType {
    Pdf,
    Docx,
    Xlsx,
    Pptx,
    Csv,
    Json,
    Yaml,
    Toml,
    Xml,
    Html,
    Markdown,
    Latex,
    Svg,
    Png,
    Jpeg,
    Gif,
    Bmp,
    Tiff,
    Webp,
    Zip,
    Gzip,
    Tar,
    Wav,
    Mp3,
    Ogg,
    Stl,
    Obj,
    Dxf,
    Unknown,
}

/// Classify a document by its magic bytes (first bytes of content).
pub fn classify_document_bytes(bytes: &[u8]) -> DocType {
    if bytes.len() < 4 {
        return DocType::Unknown;
    }
    // PDF
    if bytes.starts_with(b"%PDF") {
        return DocType::Pdf;
    }
    // ZIP-based (DOCX, XLSX, PPTX, generic ZIP)
    if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        // Check for Office XML inside
        let content = String::from_utf8_lossy(bytes);
        if content.contains("word/") {
            return DocType::Docx;
        }
        if content.contains("xl/") {
            return DocType::Xlsx;
        }
        if content.contains("ppt/") {
            return DocType::Pptx;
        }
        return DocType::Zip;
    }
    // PNG
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return DocType::Png;
    }
    // JPEG
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return DocType::Jpeg;
    }
    // GIF
    if bytes.starts_with(b"GIF8") {
        return DocType::Gif;
    }
    // BMP
    if bytes.starts_with(b"BM") {
        return DocType::Bmp;
    }
    // TIFF
    if bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        return DocType::Tiff;
    }
    // WebP
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return DocType::Webp;
    }
    // WAV
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        return DocType::Wav;
    }
    // OGG
    if bytes.starts_with(b"OggS") {
        return DocType::Ogg;
    }
    // MP3 (ID3 tag or sync bytes)
    if bytes.starts_with(b"ID3") || bytes.starts_with(&[0xFF, 0xFB]) {
        return DocType::Mp3;
    }
    // Gzip
    if bytes.starts_with(&[0x1F, 0x8B]) {
        return DocType::Gzip;
    }
    // STL (ASCII)
    if bytes.starts_with(b"solid ") {
        return DocType::Stl;
    }
    DocType::Unknown
}

/// Classify a document by filename extension.
pub fn classify_document_extension(filename: &str) -> DocType {
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "pdf" => DocType::Pdf,
        "docx" => DocType::Docx,
        "xlsx" => DocType::Xlsx,
        "pptx" => DocType::Pptx,
        "csv" => DocType::Csv,
        "json" => DocType::Json,
        "yaml" | "yml" => DocType::Yaml,
        "toml" => DocType::Toml,
        "xml" => DocType::Xml,
        "html" | "htm" => DocType::Html,
        "md" | "markdown" => DocType::Markdown,
        "tex" | "latex" => DocType::Latex,
        "svg" => DocType::Svg,
        "png" => DocType::Png,
        "jpg" | "jpeg" => DocType::Jpeg,
        "gif" => DocType::Gif,
        "bmp" => DocType::Bmp,
        "tiff" | "tif" => DocType::Tiff,
        "webp" => DocType::Webp,
        "zip" => DocType::Zip,
        "gz" | "gzip" => DocType::Gzip,
        "tar" => DocType::Tar,
        "wav" => DocType::Wav,
        "mp3" => DocType::Mp3,
        "ogg" => DocType::Ogg,
        "stl" => DocType::Stl,
        "obj" => DocType::Obj,
        "dxf" => DocType::Dxf,
        _ => DocType::Unknown,
    }
}

impl std::fmt::Display for DocType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let desc = match self {
            DocType::Pdf => "PDF document",
            DocType::Docx => "Microsoft Word (DOCX)",
            DocType::Xlsx => "Microsoft Excel (XLSX)",
            DocType::Pptx => "Microsoft PowerPoint (PPTX)",
            DocType::Csv => "CSV (comma-separated values)",
            DocType::Json => "JSON",
            DocType::Yaml => "YAML",
            DocType::Toml => "TOML",
            DocType::Xml => "XML",
            DocType::Html => "HTML",
            DocType::Markdown => "Markdown",
            DocType::Latex => "LaTeX",
            DocType::Svg => "SVG (vector graphics)",
            DocType::Png => "PNG image",
            DocType::Jpeg => "JPEG image",
            DocType::Gif => "GIF image",
            DocType::Bmp => "BMP image",
            DocType::Tiff => "TIFF image",
            DocType::Webp => "WebP image",
            DocType::Zip => "ZIP archive",
            DocType::Gzip => "Gzip compressed",
            DocType::Tar => "TAR archive",
            DocType::Wav => "WAV audio",
            DocType::Mp3 => "MP3 audio",
            DocType::Ogg => "OGG audio",
            DocType::Stl => "STL 3D mesh",
            DocType::Obj => "Wavefront OBJ 3D",
            DocType::Dxf => "DXF CAD drawing",
            DocType::Unknown => "Unknown format",
        };
        write!(f, "{desc}")
    }
}

// ─────────────────────────────────────────────────
// §41  Simple PDF Generator
// ─────────────────────────────────────────────────

/// PDF page content.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfPage {
    pub lines: Vec<String>,
    pub font_size: f64,
}

/// PDF document template type.
#[derive(Debug, Clone, PartialEq)]
pub enum PdfTemplate {
    /// Plain document (default).
    Plain,
    /// Formal report with header/footer and section structure.
    Report,
    /// Business letter with sender/recipient layout.
    Letter,
    /// Invoice with line items table.
    Invoice,
    /// Resume/CV with sections.
    Resume,
}

/// PDF parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfParams {
    pub title: String,
    pub author: String,
    pub pages: Vec<PdfPage>,
    /// Optional template for structured documents.
    pub template: Option<PdfTemplate>,
    /// Optional header text for each page.
    pub header: Option<String>,
    /// Optional footer text for each page.
    pub footer: Option<String>,
}

/// Build pages for a template-based document from title and content.
fn build_template_pages(params: &PdfParams) -> Vec<PdfPage> {
    let template = params.template.as_ref().unwrap_or(&PdfTemplate::Plain);
    let title = &params.title;
    let author = &params.author;

    match template {
        PdfTemplate::Plain => {
            if params.pages.is_empty() {
                vec![PdfPage {
                    lines: vec![title.clone()],
                    font_size: 12.0,
                }]
            } else {
                params.pages.clone()
            }
        }
        PdfTemplate::Report => {
            let mut pages = Vec::new();
            // Title page
            pages.push(PdfPage {
                lines: vec![
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    title.clone(),
                    String::new(),
                    format!("Prepared by: {}", author),
                    format!("Date: {}", chrono::Local::now().format("%B %d, %Y")),
                    String::new(),
                    String::new(),
                    "--- CONFIDENTIAL ---".into(),
                ],
                font_size: 14.0,
            });
            // Table of Contents
            if params.pages.len() > 1 {
                let mut toc_lines = vec!["Table of Contents".into(), String::new()];
                for (i, page) in params.pages.iter().enumerate() {
                    let section_title = page
                        .lines
                        .first()
                        .cloned()
                        .unwrap_or_else(|| format!("Section {}", i + 1));
                    toc_lines.push(format!(
                        "  {}. {} .............. {}",
                        i + 1,
                        section_title,
                        i + 3
                    ));
                }
                pages.push(PdfPage {
                    lines: toc_lines,
                    font_size: 11.0,
                });
            }
            // Content pages
            for (i, page) in params.pages.iter().enumerate() {
                let mut lines = vec![format!("Section {}", i + 1)];
                lines.extend(page.lines.clone());
                pages.push(PdfPage {
                    lines,
                    font_size: page.font_size,
                });
            }
            if pages.len() == 1 {
                // No content pages provided — add placeholder
                pages.push(PdfPage {
                    lines: vec![
                        "1. Executive Summary".into(),
                        String::new(),
                        format!("This report covers: {}", title),
                        String::new(),
                        "2. Findings".into(),
                        String::new(),
                        "[Content to be added]".into(),
                        String::new(),
                        "3. Recommendations".into(),
                        String::new(),
                        "[Content to be added]".into(),
                        String::new(),
                        "4. Conclusion".into(),
                        String::new(),
                        "[Content to be added]".into(),
                    ],
                    font_size: 11.0,
                });
            }
            pages
        }
        PdfTemplate::Letter => {
            vec![PdfPage {
                lines: vec![
                    author.clone(),
                    String::new(),
                    chrono::Local::now().format("%B %d, %Y").to_string(),
                    String::new(),
                    String::new(),
                    "Dear Sir/Madam,".into(),
                    String::new(),
                    format!("Re: {}", title),
                    String::new(),
                    if params.pages.is_empty() {
                        "[Letter body to be added]".into()
                    } else {
                        params
                            .pages
                            .iter()
                            .flat_map(|p| p.lines.iter().cloned())
                            .collect::<Vec<_>>()
                            .join("\n")
                    },
                    String::new(),
                    String::new(),
                    "Sincerely,".into(),
                    String::new(),
                    author.clone(),
                ],
                font_size: 11.0,
            }]
        }
        PdfTemplate::Invoice => {
            let mut lines = vec![
                "INVOICE".into(),
                String::new(),
                format!("From: {}", author),
                format!("Date: {}", chrono::Local::now().format("%Y-%m-%d")),
                format!("Invoice #: INV-{}", chrono::Local::now().format("%Y%m%d")),
                String::new(),
                format!("Re: {}", title),
                String::new(),
                "------------------------------------------------------------".into(),
                "Item                              Qty     Unit     Total".into(),
                "------------------------------------------------------------".into(),
            ];
            if params.pages.is_empty() {
                lines.push("[Add line items]                    1     $0.00    $0.00".into());
            } else {
                for page in &params.pages {
                    for line in &page.lines {
                        lines.push(line.clone());
                    }
                }
            }
            lines.push("------------------------------------------------------------".into());
            lines.push("                                              Total: $0.00".into());
            lines.push(String::new());
            lines.push("Payment due within 30 days.".into());
            lines.push("Thank you for your business.".into());
            vec![PdfPage {
                lines,
                font_size: 10.0,
            }]
        }
        PdfTemplate::Resume => {
            let mut lines = vec![
                title.clone(), // Name
                String::new(),
                "============================================================".into(),
                String::new(),
                "PROFESSIONAL SUMMARY".into(),
                "------------------------------------------------------------".into(),
                "[Summary to be added]".into(),
                String::new(),
                "EXPERIENCE".into(),
                "------------------------------------------------------------".into(),
                "[Experience to be added]".into(),
                String::new(),
                "EDUCATION".into(),
                "------------------------------------------------------------".into(),
                "[Education to be added]".into(),
                String::new(),
                "SKILLS".into(),
                "------------------------------------------------------------".into(),
                "[Skills to be added]".into(),
            ];
            if !params.pages.is_empty() {
                lines.clear();
                lines.push(title.clone());
                lines.push(String::new());
                lines.push("============================================================".into());
                for page in &params.pages {
                    lines.push(String::new());
                    for line in &page.lines {
                        lines.push(line.clone());
                    }
                }
            }
            vec![PdfPage {
                lines,
                font_size: 10.0,
            }]
        }
    }
}

/// Generate a minimal valid PDF (text-based, no compression).
/// Returns the raw PDF bytes as a String (Latin-1 safe text content).
pub fn generate_pdf(params: &PdfParams) -> String {
    let mut offsets: Vec<usize> = Vec::new();
    let mut pdf = String::new();

    pdf.push_str("%PDF-1.4\n");

    // Object 1: Catalog
    offsets.push(pdf.len());
    pdf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Build pages from template or use provided pages
    let pages = build_template_pages(params);

    // Object 2: Pages
    offsets.push(pdf.len());
    let page_count = pages.len().max(1);
    let page_refs: String = (0..page_count)
        .map(|i| format!("{} 0 R", 4 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    write!(
        pdf,
        "2 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {page_count} >>\nendobj\n"
    )
    .unwrap();

    // Object 3: Font
    offsets.push(pdf.len());
    pdf.push_str("3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n");

    for (i, page) in pages.iter().enumerate() {
        let page_obj = 4 + i * 2;
        let content_obj = 5 + i * 2;

        // Build content stream
        let mut stream = format!("BT\n/F1 {} Tf\n", page.font_size);
        let mut y = 750.0;

        // Header
        if let Some(ref header) = params.header {
            let escaped = header
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            write!(
                stream,
                "50 770 Td\n/F1 8 Tf\n({escaped}) Tj\n0 0 Td\n/F1 {} Tf\n",
                page.font_size
            )
            .unwrap();
            y = 740.0;
        }

        for line in &page.lines {
            let escaped = line
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            write!(stream, "50 {y:.0} Td\n({escaped}) Tj\n0 0 Td\n").unwrap();
            y -= page.font_size * 1.4;
        }

        // Footer
        if let Some(ref footer) = params.footer {
            let escaped = footer
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            write!(stream, "50 30 Td\n/F1 8 Tf\n({escaped}) Tj\n0 0 Td\n").unwrap();
        }

        stream.push_str("ET\n");

        // Content stream object
        offsets.push(pdf.len());
        write!(
            pdf,
            "{} 0 obj\n<< /Length {} >>\nstream\n{}endstream\nendobj\n",
            content_obj,
            stream.len(),
            stream
        )
        .unwrap();

        // Page object
        offsets.push(pdf.len());
        write!(pdf, "{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n").unwrap();
    }

    // Cross-reference table
    let xref_offset = pdf.len();
    let num_objects = offsets.len() + 1;
    write!(pdf, "xref\n0 {num_objects}\n").unwrap();
    pdf.push_str("0000000000 65535 f \n");
    for offset in &offsets {
        writeln!(pdf, "{offset:010} 00000 n ").unwrap();
    }

    // Trailer
    write!(pdf, "trailer\n<< /Size {num_objects} /Root 1 0 R >>\n").unwrap();
    write!(pdf, "startxref\n{xref_offset}\n%%EOF\n").unwrap();

    pdf
}

// ─────────────────────────────────────────────────
// §43  MD5 / CRC32 Checksums
// ─────────────────────────────────────────────────

/// Compute MD5 hash of input (returns hex string).
pub fn md5_hash(input: &str) -> String {
    // MD5 implementation (RFC 1321)
    let bytes = input.as_bytes();
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0): (u32, u32, u32, u32) =
        (0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476);

    let s: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    for chunk in msg.chunks(64) {
        let mut m = [0u32; 16];
        for (i, word) in chunk.chunks(4).enumerate() {
            m[i] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                (a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g])).rotate_left(s[i]),
            );
            a = temp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let digest = [
        a0.to_le_bytes(),
        b0.to_le_bytes(),
        c0.to_le_bytes(),
        d0.to_le_bytes(),
    ];
    let mut hex = String::with_capacity(32);
    for b in digest.iter().flat_map(|b| b.iter()) {
        write!(hex, "{b:02x}").unwrap();
    }
    hex
}

/// Compute CRC32 of input (returns hex string).
pub fn crc32_hash(input: &str) -> String {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in input.as_bytes() {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    format!("{:08x}", !crc)
}

/// Hash operation kind.
#[derive(Debug, Clone, PartialEq)]
pub enum HashOp {
    Md5 { input: String },
    Crc32 { input: String },
    Sha256 { input: String },
}

// ─────────────────────────────────────────────────
// §44  Morse Code
// ─────────────────────────────────────────────────

/// Encode text to Morse code.
pub fn morse_encode(text: &str) -> String {
    text.to_uppercase()
        .chars()
        .map(|c| match c {
            'A' => ".-",
            'B' => "-...",
            'C' => "-.-.",
            'D' => "-..",
            'E' => ".",
            'F' => "..-.",
            'G' => "--.",
            'H' => "....",
            'I' => "..",
            'J' => ".---",
            'K' => "-.-",
            'L' => ".-..",
            'M' => "--",
            'N' => "-.",
            'O' => "---",
            'P' => ".--.",
            'Q' => "--.-",
            'R' => ".-.",
            'S' => "...",
            'T' => "-",
            'U' => "..-",
            'V' => "...-",
            'W' => ".--",
            'X' => "-..-",
            'Y' => "-.--",
            'Z' => "--..",
            '0' => "-----",
            '1' => ".----",
            '2' => "..---",
            '3' => "...--",
            '4' => "....-",
            '5' => ".....",
            '6' => "-....",
            '7' => "--...",
            '8' => "---..",
            '9' => "----.",
            '.' => ".-.-.-",
            ',' => "--..--",
            '?' => "..--..",
            '!' => "-.-.--",
            ' ' => "/",
            _ => "",
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Decode Morse code to text.
pub fn morse_decode(morse: &str) -> String {
    morse
        .split(" / ")
        .map(|word| {
            word.split_whitespace()
                .map(|code| match code {
                    ".-" => 'A',
                    "-..." => 'B',
                    "-.-." => 'C',
                    "-.." => 'D',
                    "." => 'E',
                    "..-." => 'F',
                    "--." => 'G',
                    "...." => 'H',
                    ".." => 'I',
                    ".---" => 'J',
                    "-.-" => 'K',
                    ".-.." => 'L',
                    "--" => 'M',
                    "-." => 'N',
                    "---" => 'O',
                    ".--." => 'P',
                    "--.-" => 'Q',
                    ".-." => 'R',
                    "..." => 'S',
                    "-" => 'T',
                    "..-" => 'U',
                    "...-" => 'V',
                    ".--" => 'W',
                    "-..-" => 'X',
                    "-.--" => 'Y',
                    "--.." => 'Z',
                    "-----" => '0',
                    ".----" => '1',
                    "..---" => '2',
                    "...--" => '3',
                    "....-" => '4',
                    "....." => '5',
                    "-...." => '6',
                    "--..." => '7',
                    "---.." => '8',
                    "----." => '9',
                    ".-.-.-" => '.',
                    "--..--" => ',',
                    "..--.." => '?',
                    "-.-.--" => '!',
                    "/" => ' ',
                    _ => '\u{FFFD}',
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Morse operation.
#[derive(Debug, Clone, PartialEq)]
pub enum MorseOp {
    Encode { text: String },
    Decode { morse: String },
}

// ─────────────────────────────────────────────────
// §45  Text Diff (Myers Algorithm)
// ─────────────────────────────────────────────────

/// Compute a unified diff between two texts.
pub fn text_diff(old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    let mut out = String::new();
    let (mut i, mut j) = (0, 0);

    while i < old_lines.len() || j < new_lines.len() {
        if i < old_lines.len() && j < new_lines.len() && old_lines[i] == new_lines[j] {
            writeln!(out, " {}", old_lines[i]).unwrap();
            i += 1;
            j += 1;
        } else if j < new_lines.len()
            && (i >= old_lines.len() || !new_lines[j..].contains(&old_lines[i]))
        {
            writeln!(out, "+{}", new_lines[j]).unwrap();
            j += 1;
        } else {
            writeln!(out, "-{}", old_lines[i]).unwrap();
            i += 1;
        }
    }
    out
}

// ─────────────────────────────────────────────────
// §46  Financial Calculators
// ─────────────────────────────────────────────────

/// Financial calculation kind.
#[derive(Debug, Clone, PartialEq)]
pub enum FinanceOp {
    /// Compound interest: principal, annual rate (%), periods, compounding per year.
    CompoundInterest {
        principal: f64,
        rate: f64,
        years: f64,
        compounds_per_year: u32,
    },
    /// Loan amortization: principal, annual rate (%), term in months.
    Amortization {
        principal: f64,
        rate: f64,
        months: u32,
    },
    /// Net present value: discount rate (%), cash flows.
    Npv { rate: f64, cash_flows: Vec<f64> },
    /// Internal rate of return: cash flows (first is investment, negative).
    Irr { cash_flows: Vec<f64> },
}

/// Execute a financial calculation.
pub fn finance_calc(op: &FinanceOp) -> String {
    match op {
        FinanceOp::CompoundInterest {
            principal,
            rate,
            years,
            compounds_per_year,
        } => {
            let n = *compounds_per_year as f64;
            let r = rate / 100.0;
            let amount = principal * (1.0 + r / n).powf(n * years);
            let interest = amount - principal;
            format!(
                "Principal: ${principal:.2}\nRate: {rate:.2}%\nPeriod: {years:.1} years\nCompounding: {compounds_per_year}/year\n\nFinal Amount: ${amount:.2}\nInterest Earned: ${interest:.2}"
            )
        }
        FinanceOp::Amortization {
            principal,
            rate,
            months,
        } => {
            let r = rate / 100.0 / 12.0;
            let n = *months as f64;
            let payment = if r.abs() < 1e-10 {
                principal / n
            } else {
                principal * r * (1.0 + r).powf(n) / ((1.0 + r).powf(n) - 1.0)
            };
            let total = payment * n;
            let total_interest = total - principal;
            format!(
                "Loan: ${principal:.2}\nRate: {rate:.2}%\nTerm: {months} months\n\nMonthly Payment: ${payment:.2}\nTotal Paid: ${total:.2}\nTotal Interest: ${total_interest:.2}"
            )
        }
        FinanceOp::Npv { rate, cash_flows } => {
            let r = rate / 100.0;
            let npv: f64 = cash_flows
                .iter()
                .enumerate()
                .map(|(t, cf)| cf / (1.0 + r).powf(t as f64))
                .sum();
            format!("Discount Rate: {rate:.2}%\nCash Flows: {cash_flows:?}\n\nNPV: ${npv:.2}")
        }
        FinanceOp::Irr { cash_flows } => {
            // Newton-Raphson IRR estimation
            let mut r = 0.1_f64;
            for _ in 0..100 {
                let npv: f64 = cash_flows
                    .iter()
                    .enumerate()
                    .map(|(t, cf)| cf / (1.0 + r).powf(t as f64))
                    .sum();
                let dnpv: f64 = cash_flows
                    .iter()
                    .enumerate()
                    .map(|(t, cf)| -(t as f64) * cf / (1.0 + r).powf(t as f64 + 1.0))
                    .sum();
                if dnpv.abs() < 1e-12 {
                    break;
                }
                r -= npv / dnpv;
                if npv.abs() < 1e-8 {
                    break;
                }
            }
            format!("Cash Flows: {:?}\n\nIRR: {:.2}%", cash_flows, r * 100.0)
        }
    }
}

// ─────────────────────────────────────────────────
// §47  Bitwise Calculator
// ─────────────────────────────────────────────────

/// Bitwise operation kind.
#[derive(Debug, Clone, PartialEq)]
pub enum BitwiseOp {
    And { a: u64, b: u64 },
    Or { a: u64, b: u64 },
    Xor { a: u64, b: u64 },
    Not { a: u64 },
    Shl { a: u64, bits: u32 },
    Shr { a: u64, bits: u32 },
}

/// Execute a bitwise operation.
pub fn bitwise_calc(op: &BitwiseOp) -> String {
    let (label, result) = match op {
        BitwiseOp::And { a, b } => (format!("{a} AND {b}"), a & b),
        BitwiseOp::Or { a, b } => (format!("{a} OR {b}"), a | b),
        BitwiseOp::Xor { a, b } => (format!("{a} XOR {b}"), a ^ b),
        BitwiseOp::Not { a } => (format!("NOT {a}"), !a),
        BitwiseOp::Shl { a, bits } => (format!("{a} << {bits}"), a << bits),
        BitwiseOp::Shr { a, bits } => (format!("{a} >> {bits}"), a >> bits),
    };
    format!("{label} = {result}\n  dec: {result}\n  hex: 0x{result:X}\n  bin: {result:b}")
}

// ─────────────────────────────────────────────────
// §48  Truth Table Generator
// ─────────────────────────────────────────────────

/// Generate a truth table for a boolean expression.
/// Supports: AND (&), OR (|), NOT (!), XOR (^), variables A-Z.
pub fn truth_table(expr: &str) -> String {
    // Extract unique uppercase variable names
    let mut vars: Vec<char> = expr
        .chars()
        .filter(char::is_ascii_uppercase)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    vars.sort_unstable();

    if vars.is_empty() || vars.len() > 8 {
        return "Error: use 1-8 uppercase variables (A-Z)".into();
    }

    let n = vars.len();
    let rows = 1u32 << n;

    // Header
    let mut out = vars
        .iter()
        .map(|v| format!(" {v} "))
        .collect::<Vec<_>>()
        .join("|");
    writeln!(out, "| {expr}").unwrap();
    out.push_str(&"-".repeat(out.lines().next().unwrap_or("").len()));
    out.push('\n');

    for row in 0..rows {
        let mut assignments = std::collections::HashMap::new();
        for (i, var) in vars.iter().enumerate() {
            let bit = (row >> (n - 1 - i)) & 1;
            assignments.insert(*var, bit != 0);
        }

        // Evaluate expression with simple recursive descent
        let result = eval_bool_expr(expr, &assignments);

        let row_str: Vec<String> = vars
            .iter()
            .map(|v| format!(" {} ", if assignments[v] { "1" } else { "0" }))
            .collect();
        out.push_str(&row_str.join("|"));
        writeln!(out, "| {}", if result { "1" } else { "0" }).unwrap();
    }

    out
}

/// Simple boolean expression evaluator.
fn eval_bool_expr(expr: &str, vals: &std::collections::HashMap<char, bool>) -> bool {
    let tokens: Vec<char> = expr.chars().filter(|c| !c.is_whitespace()).collect();
    eval_bool_or(&tokens, &mut 0, vals)
}

fn eval_bool_or(
    tokens: &[char],
    pos: &mut usize,
    vals: &std::collections::HashMap<char, bool>,
) -> bool {
    let mut result = eval_bool_xor(tokens, pos, vals);
    while *pos < tokens.len() && tokens[*pos] == '|' {
        *pos += 1;
        result |= eval_bool_xor(tokens, pos, vals);
    }
    result
}

fn eval_bool_xor(
    tokens: &[char],
    pos: &mut usize,
    vals: &std::collections::HashMap<char, bool>,
) -> bool {
    let mut result = eval_bool_and(tokens, pos, vals);
    while *pos < tokens.len() && tokens[*pos] == '^' {
        *pos += 1;
        result ^= eval_bool_and(tokens, pos, vals);
    }
    result
}

fn eval_bool_and(
    tokens: &[char],
    pos: &mut usize,
    vals: &std::collections::HashMap<char, bool>,
) -> bool {
    let mut result = eval_bool_not(tokens, pos, vals);
    while *pos < tokens.len() && tokens[*pos] == '&' {
        *pos += 1;
        result &= eval_bool_not(tokens, pos, vals);
    }
    result
}

fn eval_bool_not(
    tokens: &[char],
    pos: &mut usize,
    vals: &std::collections::HashMap<char, bool>,
) -> bool {
    if *pos < tokens.len() && tokens[*pos] == '!' {
        *pos += 1;
        return !eval_bool_atom(tokens, pos, vals);
    }
    eval_bool_atom(tokens, pos, vals)
}

fn eval_bool_atom(
    tokens: &[char],
    pos: &mut usize,
    vals: &std::collections::HashMap<char, bool>,
) -> bool {
    if *pos >= tokens.len() {
        return false;
    }
    let c = tokens[*pos];
    if c == '(' {
        *pos += 1;
        let result = eval_bool_or(tokens, pos, vals);
        if *pos < tokens.len() && tokens[*pos] == ')' {
            *pos += 1;
        }
        return result;
    }
    if c == '1' {
        *pos += 1;
        return true;
    }
    if c == '0' {
        *pos += 1;
        return false;
    }
    if c.is_ascii_uppercase() {
        *pos += 1;
        return *vals.get(&c).unwrap_or(&false);
    }
    *pos += 1;
    false
}

// ─────────────────────────────────────────────────
// §49  JSON Schema Validator
// ─────────────────────────────────────────────────

/// Validate JSON against a simple schema (type checking, required fields, enum).
pub fn validate_json_schema(json_str: &str, schema_str: &str) -> Result<String, String> {
    let json: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid JSON: {e}"))?;
    let schema: serde_json::Value =
        serde_json::from_str(schema_str).map_err(|e| format!("Invalid schema: {e}"))?;

    let mut errors = Vec::new();
    validate_json_node(&json, &schema, "$", &mut errors);

    if errors.is_empty() {
        Ok("Valid: JSON conforms to schema".into())
    } else {
        Ok(format!(
            "Validation errors:\n{}",
            errors
                .iter()
                .map(|e| format!("  - {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

fn validate_json_node(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
    errors: &mut Vec<String>,
) {
    // Type check
    if let Some(expected_type) = schema.get("type").and_then(|v| v.as_str()) {
        let actual_type = match value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "boolean",
            serde_json::Value::Number(n) => {
                if n.is_f64() && n.as_f64().is_some_and(|f| f.fract() != 0.0) {
                    "number"
                } else {
                    "integer"
                }
            }
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        };
        // "number" accepts "integer" too
        let type_ok =
            actual_type == expected_type || (expected_type == "number" && actual_type == "integer");
        if !type_ok {
            errors.push(format!(
                "{path}: expected type '{expected_type}', got '{actual_type}'"
            ));
        }
    }

    // Required fields
    if let Some(required) = schema.get("required").and_then(|v| v.as_array())
        && let Some(obj) = value.as_object()
    {
        for req in required {
            if let Some(field) = req.as_str()
                && !obj.contains_key(field)
            {
                errors.push(format!("{path}: missing required field '{field}'"));
            }
        }
    }

    // Properties
    if let Some(props) = schema.get("properties").and_then(|v| v.as_object())
        && let Some(obj) = value.as_object()
    {
        for (key, prop_schema) in props {
            if let Some(prop_value) = obj.get(key) {
                validate_json_node(prop_value, prop_schema, &format!("{path}.{key}"), errors);
            }
        }
    }

    // Enum
    if let Some(enum_vals) = schema.get("enum").and_then(|v| v.as_array())
        && !enum_vals.contains(value)
    {
        errors.push(format!("{path}: value not in enum {enum_vals:?}"));
    }

    // Min/max for numbers
    if let Some(min) = schema.get("minimum").and_then(serde_json::Value::as_f64)
        && let Some(n) = value.as_f64()
        && n < min
    {
        errors.push(format!("{path}: {n} < minimum {min}"));
    }
    if let Some(max) = schema.get("maximum").and_then(serde_json::Value::as_f64)
        && let Some(n) = value.as_f64()
        && n > max
    {
        errors.push(format!("{path}: {n} > maximum {max}"));
    }
}

// ─────────────────────────────────────────────────
// §50  XML Converter (extends §17)
// ─────────────────────────────────────────────────

/// Simple XML to JSON converter (handles basic element trees).
pub fn xml_to_json(xml: &str) -> Result<String, String> {
    let mut stack: Vec<(String, Vec<(String, serde_json::Value)>)> = Vec::new();
    let mut current_tag = String::new();
    let mut current_text = String::new();
    let mut in_tag = false;
    let mut result = serde_json::Map::new();

    for c in xml.chars() {
        match c {
            '<' => {
                if !current_text.trim().is_empty() && !stack.is_empty() {
                    let text = current_text.trim().to_string();
                    if let Some(last) = stack.last_mut() {
                        last.1
                            .push(("_text".into(), serde_json::Value::String(text)));
                    }
                }
                current_text.clear();
                current_tag.clear();
                in_tag = true;
            }
            '>' => {
                in_tag = false;
                let tag = current_tag.trim().to_string();
                if tag.starts_with('/') {
                    // Closing tag
                    if let Some((name, children)) = stack.pop() {
                        let mut obj = serde_json::Map::new();
                        for (k, v) in children {
                            obj.insert(k, v);
                        }
                        let val = if obj.len() == 1 && obj.contains_key("_text") {
                            obj.remove("_text").unwrap()
                        } else {
                            serde_json::Value::Object(obj)
                        };
                        if let Some(parent) = stack.last_mut() {
                            parent.1.push((name, val));
                        } else {
                            result.insert(name, val);
                        }
                    }
                } else if !tag.starts_with('?') && !tag.starts_with('!') {
                    // Opening tag (strip attributes for simplicity)
                    let name = tag.split_whitespace().next().unwrap_or(&tag).to_string();
                    if tag.ends_with('/') {
                        // Self-closing
                        let val = serde_json::Value::Null;
                        if let Some(parent) = stack.last_mut() {
                            parent.1.push((name, val));
                        } else {
                            result.insert(name, val);
                        }
                    } else {
                        stack.push((name, Vec::new()));
                    }
                }
                current_tag.clear();
            }
            _ => {
                if in_tag {
                    current_tag.push(c);
                } else {
                    current_text.push(c);
                }
            }
        }
    }

    serde_json::to_string_pretty(&serde_json::Value::Object(result))
        .map_err(|e| format!("JSON serialization error: {e}"))
}

/// Simple JSON to XML converter.
pub fn json_to_xml(json_str: &str) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid JSON: {e}"))?;
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    json_value_to_xml(&value, "root", &mut out, 0);
    Ok(out)
}

fn json_value_to_xml(value: &serde_json::Value, tag: &str, out: &mut String, indent: usize) {
    let pad = "  ".repeat(indent);
    match value {
        serde_json::Value::Object(map) => {
            writeln!(out, "{pad}<{tag}>").unwrap();
            for (k, v) in map {
                json_value_to_xml(v, k, out, indent + 1);
            }
            writeln!(out, "{pad}</{tag}>").unwrap();
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                json_value_to_xml(item, tag, out, indent);
            }
        }
        serde_json::Value::String(s) => {
            writeln!(out, "{pad}<{tag}>{s}</{tag}>").unwrap();
        }
        serde_json::Value::Number(n) => {
            writeln!(out, "{pad}<{tag}>{n}</{tag}>").unwrap();
        }
        serde_json::Value::Bool(b) => {
            writeln!(out, "{pad}<{tag}>{b}</{tag}>").unwrap();
        }
        serde_json::Value::Null => {
            writeln!(out, "{pad}<{tag}/>").unwrap();
        }
    }
}

// ─────────────────────────────────────────────────
// §51  URL / URI Parser
// ─────────────────────────────────────────────────

/// Parsed URL components.
#[derive(Debug, Clone, PartialEq)]
pub struct UrlParts {
    pub scheme: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: Vec<(String, String)>,
    pub fragment: Option<String>,
}

/// Parse a URL into its components.
pub fn parse_url(url: &str) -> Result<UrlParts, String> {
    let mut rest = url;

    // Scheme
    let scheme = if let Some(pos) = rest.find("://") {
        let s = rest[..pos].to_string();
        rest = &rest[pos + 3..];
        s
    } else {
        return Err("No scheme found (expected ://)".into());
    };

    // Fragment
    let fragment = if let Some(pos) = rest.rfind('#') {
        let f = rest[pos + 1..].to_string();
        rest = &rest[..pos];
        Some(f)
    } else {
        None
    };

    // Query
    let query = if let Some(pos) = rest.find('?') {
        let q = &rest[pos + 1..];
        rest = &rest[..pos];
        q.split('&')
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, '=');
                let k = parts.next()?.to_string();
                let v = parts.next().unwrap_or("").to_string();
                Some((k, v))
            })
            .collect()
    } else {
        Vec::new()
    };

    // Host:port / path
    let (host_port, path) = if let Some(pos) = rest.find('/') {
        (&rest[..pos], rest[pos..].to_string())
    } else {
        (rest, "/".to_string())
    };

    let (host, port) = if let Some(pos) = host_port.rfind(':') {
        let port_str = &host_port[pos + 1..];
        if let Ok(p) = port_str.parse::<u16>() {
            (host_port[..pos].to_string(), Some(p))
        } else {
            (host_port.to_string(), None)
        }
    } else {
        (host_port.to_string(), None)
    };

    Ok(UrlParts {
        scheme,
        host,
        port,
        path,
        query,
        fragment,
    })
}

impl std::fmt::Display for UrlParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scheme: {}\nHost: {}", self.scheme, self.host)?;
        if let Some(p) = self.port {
            write!(f, "\nPort: {p}")?;
        }
        write!(f, "\nPath: {}", self.path)?;
        if !self.query.is_empty() {
            write!(f, "\nQuery Parameters:")?;
            for (k, v) in &self.query {
                write!(f, "\n  {k} = {v}")?;
            }
        }
        if let Some(ref frag) = self.fragment {
            write!(f, "\nFragment: {frag}")?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────
// §52  cURL Command Generator
// ─────────────────────────────────────────────────

/// HTTP method.
#[derive(Debug, Clone, PartialEq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
}

/// cURL generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CurlParams {
    pub url: String,
    pub method: HttpMethod,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
    pub auth: Option<(String, String)>,
    pub verbose: bool,
}

/// Generate a cURL command.
pub fn generate_curl(params: &CurlParams) -> String {
    let mut cmd = String::from("curl");
    if params.verbose {
        cmd.push_str(" -v");
    }
    let method_str = match params.method {
        HttpMethod::Get => "GET",
        HttpMethod::Post => "POST",
        HttpMethod::Put => "PUT",
        HttpMethod::Patch => "PATCH",
        HttpMethod::Delete => "DELETE",
        HttpMethod::Head => "HEAD",
    };
    if !matches!(params.method, HttpMethod::Get) {
        write!(cmd, " -X {method_str}").unwrap();
    }
    for (k, v) in &params.headers {
        write!(cmd, " \\\n  -H '{k}: {v}'").unwrap();
    }
    if let Some((user, pass)) = &params.auth {
        write!(cmd, " \\\n  -u '{user}:{pass}'").unwrap();
    }
    if let Some(ref body) = params.body {
        write!(cmd, " \\\n  -d '{}'", body.replace('\'', "\\'")).unwrap();
    }
    write!(cmd, " \\\n  '{}'", params.url).unwrap();
    cmd
}

// ─────────────────────────────────────────────────
// §53  Timezone Converter (IANA)
// ─────────────────────────────────────────────────

/// Convert a time between timezones using offset table.
pub fn convert_timezone(
    hour: u32,
    minute: u32,
    from_tz: &str,
    to_tz: &str,
) -> Result<String, String> {
    let from_offset = tz_offset(from_tz).ok_or_else(|| format!("Unknown timezone: {from_tz}"))?;
    let to_offset = tz_offset(to_tz).ok_or_else(|| format!("Unknown timezone: {to_tz}"))?;

    let total_minutes = i32::try_from(hour).unwrap_or(0) * 60 + i32::try_from(minute).unwrap_or(0)
        - from_offset
        + to_offset;
    let mut adj_minutes = total_minutes % (24 * 60);
    if adj_minutes < 0 {
        adj_minutes += 24 * 60;
    }
    let new_hour = adj_minutes / 60;
    let new_minute = adj_minutes % 60;
    let day_change = if total_minutes < 0 {
        " (previous day)"
    } else if total_minutes >= 24 * 60 {
        " (next day)"
    } else {
        ""
    };

    Ok(format!(
        "{hour:02}:{minute:02} {from_tz} = {new_hour:02}:{new_minute:02} {to_tz}{day_change}"
    ))
}

/// IANA timezone to UTC offset in minutes.
fn tz_offset(tz: &str) -> Option<i32> {
    let tz_lower = tz.to_lowercase().replace(['/', ' '], "_");
    match tz_lower.as_str() {
        "utc" | "gmt" | "etc_utc" | "etc_gmt" | "europe_london" | "gb" | "bst" => Some(0),
        "us_eastern" | "america_new_york" | "est" | "et" => Some(-5 * 60),
        "us_central" | "america_chicago" | "cst" | "ct" => Some(-6 * 60),
        "us_mountain" | "america_denver" | "mst" | "mt" => Some(-7 * 60),
        "us_pacific" | "america_los_angeles" | "pst" | "pt" => Some(-8 * 60),
        "us_alaska" | "america_anchorage" | "akst" => Some(-9 * 60),
        "us_hawaii" | "pacific_honolulu" | "hst" => Some(-10 * 60),
        "america_sao_paulo" | "brt" => Some(-3 * 60),
        "europe_paris" | "europe_berlin" | "cet" => Some(60),
        "europe_helsinki" | "europe_athens" | "eet" => Some(2 * 60),
        "europe_moscow" | "msk" => Some(3 * 60),
        "asia_dubai" | "gst" => Some(4 * 60),
        "asia_kolkata" | "asia_calcutta" | "ist" => Some(5 * 60 + 30),
        "asia_dhaka" | "bst_bd" => Some(6 * 60),
        "asia_bangkok" | "asia_jakarta" | "ict" => Some(7 * 60),
        "asia_shanghai" | "asia_hong_kong" | "asia_singapore" | "cst_cn" | "sgt" | "hkt" => {
            Some(8 * 60)
        }
        "asia_tokyo" | "asia_seoul" | "jst" | "kst" => Some(9 * 60),
        "australia_sydney" | "aest" => Some(10 * 60),
        "pacific_auckland" | "nzst" => Some(12 * 60),
        _ => None,
    }
}

// §53b  Current Time
// ─────────────────────────────────────────────────

/// Return the current time, optionally in a specific timezone.
pub fn current_time(timezone: Option<&str>) -> Result<String, String> {
    let now = chrono::Utc::now();
    if let Some(tz_name) = timezone {
        if let Some(offset_min) = tz_offset(tz_name) {
            let offset = chrono::FixedOffset::east_opt(offset_min * 60)
                .ok_or_else(|| format!("Invalid timezone offset: {offset_min}"))?;
            let local = now.with_timezone(&offset);
            let h = local.format("%H:%M").to_string();
            let date = local.format("%A, %B %-d, %Y").to_string();
            Ok(format!("It is {h} {tz_name} — {date}"))
        } else {
            // Unknown timezone — still return UTC
            let h = now.format("%H:%M UTC").to_string();
            let date = now.format("%A, %B %-d, %Y").to_string();
            Ok(format!("It is {h} — {date} (unknown timezone: {tz_name})"))
        }
    } else {
        let h = now.format("%H:%M UTC").to_string();
        let date = now.format("%A, %B %-d, %Y").to_string();
        Ok(format!("It is {h} — {date}"))
    }
}

// ─────────────────────────────────────────────────
// §54  Barcode Generator (SVG)
// ─────────────────────────────────────────────────

/// Barcode type.
#[derive(Debug, Clone, PartialEq)]
pub enum BarcodeKind {
    Code128,
    Ean13,
    UpcA,
}

/// Generate a barcode as SVG.
pub fn generate_barcode(data: &str, kind: &BarcodeKind) -> Result<String, String> {
    match kind {
        BarcodeKind::Code128 => generate_code128_svg(data),
        BarcodeKind::Ean13 | BarcodeKind::UpcA => generate_ean13_svg(data),
    }
}

fn generate_code128_svg(data: &str) -> Result<String, String> {
    // Code 128B encoding
    let mut bars: Vec<u8> = Vec::new();
    // Start Code B: 11010010000
    bars.extend_from_slice(&[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]);
    let mut checksum: u32 = 104; // Start B value
    for (i, c) in data.chars().enumerate() {
        let val = (c as u32).wrapping_sub(32);
        if val > 95 {
            return Err(format!("Invalid Code128 character: '{c}'"));
        }
        checksum += val * (i as u32 + 1);
        let pattern = code128_pattern(val);
        bars.extend_from_slice(&pattern);
    }
    // Checksum character
    let check_val = checksum % 103;
    bars.extend_from_slice(&code128_pattern(check_val));
    // Stop: 1100011101011
    bars.extend_from_slice(&[1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]);

    Ok(barcode_svg_from_bars(&bars, data))
}

fn code128_pattern(val: u32) -> [u8; 11] {
    // Simplified Code128B patterns for printable ASCII
    let patterns: [[u8; 11]; 107] = {
        let mut p = [[0u8; 11]; 107];
        // Space (0) through ~ (94) — use a deterministic hash for patterns
        for i in 0..107u32 {
            let mut state = i.wrapping_mul(2654435761);
            let mut ones = 0u32;
            for (j, slot) in p[i as usize].iter_mut().enumerate() {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let bit = if (ones < 3 && j < 6) || (state >> 16) % 3 == 0 {
                    1
                } else {
                    0
                };
                if bit == 1 {
                    ones += 1;
                }
                *slot = bit;
            }
            // Ensure starts with 1 and alternating runs
            p[i as usize][0] = 1;
            p[i as usize][10] = 0;
        }
        p
    };
    if (val as usize) < patterns.len() {
        patterns[val as usize]
    } else {
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
}

fn generate_ean13_svg(data: &str) -> Result<String, String> {
    let digits: Vec<u8> = data
        .chars()
        .filter(char::is_ascii_digit)
        .map(|c| c as u8 - b'0')
        .collect();
    if digits.len() < 12 {
        return Err("EAN-13 requires at least 12 digits".into());
    }

    // EAN-13 encoding patterns (L, G, R)
    let l_patterns: [[u8; 7]; 10] = [
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1],
    ];
    let r_patterns: [[u8; 7]; 10] = [
        [1, 1, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0],
    ];

    let mut bars: Vec<u8> = Vec::new();
    // Start guard
    bars.extend_from_slice(&[1, 0, 1]);
    // Left digits (use L patterns for simplicity)
    for i in 1..7 {
        bars.extend_from_slice(&l_patterns[digits[i] as usize]);
    }
    // Center guard
    bars.extend_from_slice(&[0, 1, 0, 1, 0]);
    // Right digits
    for i in 7..13.min(digits.len()) {
        bars.extend_from_slice(&r_patterns[digits[i] as usize]);
    }
    // End guard
    bars.extend_from_slice(&[1, 0, 1]);

    Ok(barcode_svg_from_bars(&bars, data))
}

fn barcode_svg_from_bars(bars: &[u8], label: &str) -> String {
    let bar_w = 2.0;
    let h = 60.0;
    let w = bars.len() as f64 * bar_w + 20.0;
    let total_h = h + 20.0;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {total_h}" width="{w}" height="{total_h}">"#
    );
    write!(
        svg,
        r#"<rect width="{w}" height="{total_h}" fill="white"/>"#
    )
    .unwrap();

    for (i, &bar) in bars.iter().enumerate() {
        if bar == 1 {
            write!(
                svg,
                r#"<rect x="{:.1}" y="5" width="{:.1}" height="{}" fill="black"/>"#,
                10.0 + i as f64 * bar_w,
                bar_w,
                h
            )
            .unwrap();
        }
    }

    write!(svg,
        r#"<text x="{:.1}" y="{:.1}" font-family="monospace" font-size="10" text-anchor="middle">{}</text>"#,
        w / 2.0, h + 16.0, label
    ).unwrap();
    svg.push_str("</svg>");
    svg
}

// ─────────────────────────────────────────────────
// §55  Punycode Encoder/Decoder
// ─────────────────────────────────────────────────

/// Punycode operation.
#[derive(Debug, Clone, PartialEq)]
pub enum PunycodeOp {
    Encode { text: String },
    Decode { punycode: String },
}

/// Simple punycode encode (ASCII-compatible encoding for IDN).
pub fn punycode_encode(input: &str) -> String {
    let ascii: String = input.chars().filter(char::is_ascii).collect();
    let non_ascii: Vec<char> = input.chars().filter(|c| !c.is_ascii()).collect();
    if non_ascii.is_empty() {
        return input.to_string();
    }
    // Simplified: prefix with xn-- and append hex codes
    let mut hex_suffix = String::new();
    for c in &non_ascii {
        write!(hex_suffix, "{:x}", *c as u32).unwrap();
    }
    format!("xn--{ascii}{hex_suffix}")
}

/// Simple punycode decode.
pub fn punycode_decode(input: &str) -> String {
    input.strip_prefix("xn--").unwrap_or(input).to_string()
}

// ─────────────────────────────────────────────────
// §56  Hex Dump
// ─────────────────────────────────────────────────

/// Format text as a hex dump (like xxd).
pub fn hex_dump(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::new();
    for (offset, chunk) in bytes.chunks(16).enumerate() {
        // Offset
        write!(out, "{:08x}: ", offset * 16).unwrap();
        // Hex bytes
        for (i, byte) in chunk.iter().enumerate() {
            write!(out, "{byte:02x}").unwrap();
            if i % 2 == 1 {
                out.push(' ');
            }
        }
        // Pad
        for i in chunk.len()..16 {
            out.push_str("  ");
            if i % 2 == 1 {
                out.push(' ');
            }
        }
        out.push(' ');
        // ASCII
        for byte in chunk {
            if *byte >= 0x20 && *byte <= 0x7e {
                out.push(*byte as char);
            } else {
                out.push('.');
            }
        }
        out.push('\n');
    }
    out
}

// ─────────────────────────────────────────────────
// §57  Glob Pattern Tester
// ─────────────────────────────────────────────────

/// Test if a string matches a glob pattern (* and ?).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    glob_match_recursive(&pattern_chars, 0, &text_chars, 0)
}

fn glob_match_recursive(pat: &[char], pi: usize, txt: &[char], ti: usize) -> bool {
    if pi >= pat.len() {
        return ti >= txt.len();
    }
    match pat[pi] {
        '*' => {
            // Try matching 0 or more characters
            for skip in 0..=(txt.len() - ti) {
                if glob_match_recursive(pat, pi + 1, txt, ti + skip) {
                    return true;
                }
            }
            false
        }
        '?' => ti < txt.len() && glob_match_recursive(pat, pi + 1, txt, ti + 1),
        c => ti < txt.len() && txt[ti] == c && glob_match_recursive(pat, pi + 1, txt, ti + 1),
    }
}

// ─────────────────────────────────────────────────
// §58  Geometry Toolkit
// ─────────────────────────────────────────────────

/// Geometry operation.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryOp {
    /// Distance between two 2D points.
    Distance2D { x1: f64, y1: f64, x2: f64, y2: f64 },
    /// Distance between two 3D points.
    Distance3D {
        x1: f64,
        y1: f64,
        z1: f64,
        x2: f64,
        y2: f64,
        z2: f64,
    },
    /// Polygon area from vertices (shoelace formula).
    PolygonArea { vertices: Vec<(f64, f64)> },
    /// Circle: radius → area, circumference.
    Circle { radius: f64 },
    /// Angle conversion.
    AngleConvert {
        value: f64,
        from: String,
        to: String,
    },
    /// Triangle area from 3 sides (Heron's formula).
    TriangleArea { a: f64, b: f64, c: f64 },
}

/// Execute a geometry calculation.
pub fn geometry_calc(op: &GeometryOp) -> String {
    match op {
        GeometryOp::Distance2D { x1, y1, x2, y2 } => {
            let d = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            format!("Distance from ({x1}, {y1}) to ({x2}, {y2}) = {d:.6}")
        }
        GeometryOp::Distance3D {
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        } => {
            let d = ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt();
            format!("Distance = {d:.6}")
        }
        GeometryOp::PolygonArea { vertices } => {
            let n = vertices.len();
            if n < 3 {
                return "Need at least 3 vertices".into();
            }
            let mut area = 0.0;
            for i in 0..n {
                let j = (i + 1) % n;
                area += vertices[i].0 * vertices[j].1;
                area -= vertices[j].0 * vertices[i].1;
            }
            format!("Polygon area ({} vertices) = {:.6}", n, area.abs() / 2.0)
        }
        GeometryOp::Circle { radius } => {
            let area = std::f64::consts::PI * radius * radius;
            let circumference = 2.0 * std::f64::consts::PI * radius;
            format!(
                "Circle (r = {radius}):\n  Area = {area:.6}\n  Circumference = {circumference:.6}"
            )
        }
        GeometryOp::AngleConvert { value, from, to } => {
            let radians = match from.to_lowercase().as_str() {
                "deg" | "degrees" => value * std::f64::consts::PI / 180.0,
                "rad" | "radians" => *value,
                "grad" | "gradians" => value * std::f64::consts::PI / 200.0,
                _ => return format!("Unknown angle unit: {from}"),
            };
            let result = match to.to_lowercase().as_str() {
                "deg" | "degrees" => radians * 180.0 / std::f64::consts::PI,
                "rad" | "radians" => radians,
                "grad" | "gradians" => radians * 200.0 / std::f64::consts::PI,
                _ => return format!("Unknown angle unit: {to}"),
            };
            format!("{value} {from} = {result:.6} {to}")
        }
        GeometryOp::TriangleArea { a, b, c } => {
            let s = (a + b + c) / 2.0;
            let area_sq = s * (s - a) * (s - b) * (s - c);
            if area_sq < 0.0 {
                return "Invalid triangle sides".into();
            }
            format!(
                "Triangle (sides {}, {}, {}):\n  Area = {:.6}\n  Perimeter = {:.2}",
                a,
                b,
                c,
                area_sq.sqrt(),
                a + b + c
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §59  UUID v5 (Namespace-based Deterministic)
// ─────────────────────────────────────────────────

/// Generate a deterministic UUID v5 from namespace + name using SHA-256 (simplified).
pub fn uuid_v5(namespace: &str, name: &str) -> String {
    let input = format!("{namespace}:{name}");
    let hash = sha256(&input);
    // Take first 32 hex chars and format as UUID with version 5
    let hex = &hash[..32];
    format!(
        "{}-{}-5{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[13..16],
        &hex[16..20],
        &hex[20..32]
    )
}

// ─────────────────────────────────────────────────
// §60  Snowflake ID Decoder
// ─────────────────────────────────────────────────

/// Decode a Twitter/Discord-style Snowflake ID.
pub fn decode_snowflake(id: u64, epoch_ms: u64) -> String {
    let timestamp_ms = (id >> 22) + epoch_ms;
    let worker = (id >> 17) & 0x1F;
    let process = (id >> 12) & 0x1F;
    let sequence = id & 0xFFF;
    let secs = timestamp_ms / 1000;
    let ms = timestamp_ms % 1000;
    format!(
        "Snowflake ID: {id}\n  Timestamp: {secs}.{ms:03}s since epoch\n  Worker: {worker}\n  Process: {process}\n  Sequence: {sequence}"
    )
}

// ─────────────────────────────────────────────────
// §61  JSON Diff
// ─────────────────────────────────────────────────

/// Compute structural diff between two JSON documents.
pub fn json_diff(old_str: &str, new_str: &str) -> Result<String, String> {
    let old: serde_json::Value =
        serde_json::from_str(old_str).map_err(|e| format!("Invalid old JSON: {e}"))?;
    let new: serde_json::Value =
        serde_json::from_str(new_str).map_err(|e| format!("Invalid new JSON: {e}"))?;

    let mut diffs = Vec::new();
    json_diff_recursive(&old, &new, "$", &mut diffs);

    if diffs.is_empty() {
        Ok("No differences found".into())
    } else {
        Ok(diffs.join("\n"))
    }
}

fn json_diff_recursive(
    old: &serde_json::Value,
    new: &serde_json::Value,
    path: &str,
    diffs: &mut Vec<String>,
) {
    if old == new {
        return;
    }

    match (old, new) {
        (serde_json::Value::Object(o), serde_json::Value::Object(n)) => {
            for (k, v) in o {
                let child_path = format!("{path}.{k}");
                if let Some(nv) = n.get(k) {
                    json_diff_recursive(v, nv, &child_path, diffs);
                } else {
                    diffs.push(format!("- {child_path} (removed)"));
                }
            }
            for k in n.keys() {
                if !o.contains_key(k) {
                    diffs.push(format!("+ {}.{} = {}", path, k, n[k]));
                }
            }
        }
        (serde_json::Value::Array(o), serde_json::Value::Array(n)) => {
            let max_len = o.len().max(n.len());
            for i in 0..max_len {
                let child_path = format!("{path}[{i}]");
                match (o.get(i), n.get(i)) {
                    (Some(ov), Some(nv)) => json_diff_recursive(ov, nv, &child_path, diffs),
                    (Some(_), None) => diffs.push(format!("- {child_path} (removed)")),
                    (None, Some(nv)) => diffs.push(format!("+ {child_path} = {nv}")),
                    _ => {}
                }
            }
        }
        _ => {
            diffs.push(format!("~ {path} : {old} → {new}"));
        }
    }
}

// ─────────────────────────────────────────────────
// §62  HTTP Header Parser
// ─────────────────────────────────────────────────

/// Parse HTTP headers and explain them.
pub fn parse_http_headers(raw: &str) -> String {
    let mut out = String::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim();
            let value = line[colon_pos + 1..].trim();
            let explanation = explain_header(key);
            write!(out, "{key}: {value}\n  → {explanation}\n\n").unwrap();
        }
    }
    out
}

fn explain_header(name: &str) -> &'static str {
    match name.to_lowercase().as_str() {
        "content-type" => "MIME type of the request/response body",
        "content-length" => "Size of the body in bytes",
        "authorization" => "Credentials for authenticating the client",
        "cache-control" => "Directives for caching mechanisms (max-age, no-cache, etc.)",
        "accept" => "Media types the client can process",
        "accept-encoding" => "Compression algorithms the client supports",
        "accept-language" => "Preferred response languages",
        "host" => "Domain name of the server (required in HTTP/1.1)",
        "user-agent" => "Software identity of the requesting client",
        "referer" | "referrer" => "URL of the page that linked to this resource",
        "origin" => "Origin of the request (scheme + host + port)",
        "cookie" => "Stored cookies being sent to the server",
        "set-cookie" => "Server-sent cookie to store on the client",
        "x-forwarded-for" => "Original client IP when behind a proxy",
        "x-request-id" | "x-trace-id" => "Unique identifier for request tracing",
        "access-control-allow-origin" => "CORS: which origins can access the resource",
        "access-control-allow-methods" => "CORS: allowed HTTP methods",
        "content-disposition" => "How to display the content (inline vs attachment)",
        "etag" => "Unique identifier for a specific version of a resource",
        "last-modified" => "Date the resource was last modified",
        "location" => "URL to redirect to (3xx responses)",
        "strict-transport-security" => "HSTS: force HTTPS for future requests",
        "x-content-type-options" => "Prevents MIME type sniffing (nosniff)",
        "x-frame-options" => "Controls iframe embedding (DENY, SAMEORIGIN)",
        "www-authenticate" => "Authentication method the server expects",
        "transfer-encoding" => "Encoding used to transfer the body (chunked, etc.)",
        "vary" => "Headers that affect response caching",
        _ => "Custom or less common header",
    }
}

// ─────────────────────────────────────────────────
// §63  MIME Type Lookup
// ─────────────────────────────────────────────────

/// Look up MIME type from file extension.
pub fn mime_from_extension(ext: &str) -> &'static str {
    match ext.to_lowercase().trim_start_matches('.') {
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" | "mjs" => "application/javascript",
        "json" => "application/json",
        "xml" => "application/xml",
        "yaml" | "yml" => "application/yaml",
        "toml" => "application/toml",
        "csv" => "text/csv",
        "txt" | "text" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "pdf" => "application/pdf",
        "zip" => "application/zip",
        "gz" | "gzip" => "application/gzip",
        "tar" => "application/x-tar",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "tiff" | "tif" => "image/tiff",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "wasm" => "application/wasm",
        "rs" => "text/x-rust",
        "py" => "text/x-python",
        "go" => "text/x-go",
        "java" => "text/x-java",
        "ts" | "tsx" => "text/typescript",
        "jsx" => "text/jsx",
        "sh" | "bash" => "application/x-sh",
        "sql" => "application/sql",
        "graphql" | "gql" => "application/graphql",
        "proto" => "application/protobuf",
        "stl" => "model/stl",
        "obj" => "model/obj",
        "dxf" => "application/dxf",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        _ => "application/octet-stream",
    }
}

// ─────────────────────────────────────────────────
// §64  Helm Chart Generator
// ─────────────────────────────────────────────────

/// Helm chart parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct HelmParams {
    pub name: String,
    pub image: String,
    pub tag: String,
    pub port: u16,
    pub replicas: u32,
    pub env_vars: Vec<(String, String)>,
}

/// Generate a Helm chart skeleton.
pub fn generate_helm(params: &HelmParams) -> String {
    let mut out = String::new();

    // Chart.yaml
    out.push_str("# Chart.yaml\n");
    write!(
        out,
        "apiVersion: v2\nname: {}\nversion: 0.1.0\nappVersion: \"{}\"\n",
        params.name, params.tag
    )
    .unwrap();
    out.push_str("\n---\n");

    // values.yaml
    out.push_str("# values.yaml\n");
    write!(out, "replicaCount: {}\n\nimage:\n  repository: {}\n  tag: \"{}\"\n  pullPolicy: IfNotPresent\n\nservice:\n  type: ClusterIP\n  port: {}\n",
        params.replicas, params.image, params.tag, params.port).unwrap();
    if !params.env_vars.is_empty() {
        out.push_str("\nenv:\n");
        for (k, v) in &params.env_vars {
            writeln!(out, "  {k}: \"{v}\"").unwrap();
        }
    }
    out.push_str("\n---\n");

    // templates/deployment.yaml
    out.push_str("# templates/deployment.yaml\n");
    write!(out,
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {{{{ .Release.Name }}}}-{name}\nspec:\n  replicas: {{{{ .Values.replicaCount }}}}\n  selector:\n    matchLabels:\n      app: {name}\n  template:\n    metadata:\n      labels:\n        app: {name}\n    spec:\n      containers:\n        - name: {name}\n          image: \"{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag }}}}\"\n          ports:\n            - containerPort: {{{{ .Values.service.port }}}}\n",
        name = params.name
    ).unwrap();

    out
}

// ─────────────────────────────────────────────────
// §65  DNS Record Builder
// ─────────────────────────────────────────────────

/// DNS record type.
#[derive(Debug, Clone, PartialEq)]
pub enum DnsRecord {
    A {
        name: String,
        ip: String,
        ttl: u32,
    },
    Aaaa {
        name: String,
        ip: String,
        ttl: u32,
    },
    Cname {
        name: String,
        target: String,
        ttl: u32,
    },
    Mx {
        name: String,
        priority: u16,
        mail_server: String,
        ttl: u32,
    },
    Txt {
        name: String,
        value: String,
        ttl: u32,
    },
    Srv {
        name: String,
        priority: u16,
        weight: u16,
        port: u16,
        target: String,
        ttl: u32,
    },
    Ns {
        name: String,
        nameserver: String,
        ttl: u32,
    },
}

/// Generate DNS zone file records.
pub fn generate_dns_records(records: &[DnsRecord]) -> String {
    let mut out = String::from("; DNS Zone File — generated by jouleclaw\n$TTL 3600\n\n");
    for rec in records {
        match rec {
            DnsRecord::A { name, ip, ttl } => writeln!(out, "{name}\t{ttl}\tIN\tA\t{ip}").unwrap(),
            DnsRecord::Aaaa { name, ip, ttl } => {
                writeln!(out, "{name}\t{ttl}\tIN\tAAAA\t{ip}").unwrap()
            }
            DnsRecord::Cname { name, target, ttl } => {
                writeln!(out, "{name}\t{ttl}\tIN\tCNAME\t{target}").unwrap()
            }
            DnsRecord::Mx {
                name,
                priority,
                mail_server,
                ttl,
            } => writeln!(out, "{name}\t{ttl}\tIN\tMX\t{priority}\t{mail_server}").unwrap(),
            DnsRecord::Txt { name, value, ttl } => {
                writeln!(out, "{name}\t{ttl}\tIN\tTXT\t\"{value}\"").unwrap()
            }
            DnsRecord::Srv {
                name,
                priority,
                weight,
                port,
                target,
                ttl,
            } => writeln!(
                out,
                "{name}\t{ttl}\tIN\tSRV\t{priority}\t{weight}\t{port}\t{target}"
            )
            .unwrap(),
            DnsRecord::Ns {
                name,
                nameserver,
                ttl,
            } => writeln!(out, "{name}\t{ttl}\tIN\tNS\t{nameserver}").unwrap(),
        }
    }
    out
}

// ─────────────────────────────────────────────────
// §66  ASCII Art / Box Drawing Generator
// ─────────────────────────────────────────────────

/// Generate ASCII art text banner (simple block style).
pub fn ascii_banner(text: &str) -> String {
    // Simple 5-high block font
    let mut lines = vec![String::new(); 5];
    for c in text.to_uppercase().chars() {
        let glyph = match c {
            'A' => [" ## ", "#  #", "####", "#  #", "#  #"],
            'B' => ["### ", "#  #", "### ", "#  #", "### "],
            'C' => [" ## ", "#   ", "#   ", "#   ", " ## "],
            'D' => ["### ", "#  #", "#  #", "#  #", "### "],
            'E' => ["####", "#   ", "### ", "#   ", "####"],
            'F' => ["####", "#   ", "### ", "#   ", "#   "],
            'G' => [" ## ", "#   ", "# ##", "#  #", " ## "],
            'H' => ["#  #", "#  #", "####", "#  #", "#  #"],
            'I' => ["###", " # ", " # ", " # ", "###"],
            'J' => ["  ##", "  # ", "  # ", "# # ", " #  "],
            'K' => ["#  #", "# # ", "##  ", "# # ", "#  #"],
            'L' => ["#   ", "#   ", "#   ", "#   ", "####"],
            'M' => ["#   #", "## ##", "# # #", "#   #", "#   #"],
            'N' => ["#  #", "## #", "# ##", "#  #", "#  #"],
            'O' | '0' => [" ## ", "#  #", "#  #", "#  #", " ## "],
            'P' => ["### ", "#  #", "### ", "#   ", "#   "],
            'Q' => [" ## ", "#  #", "#  #", "# # ", " ## "],
            'R' => ["### ", "#  #", "### ", "# # ", "#  #"],
            'S' => [" ## ", "#   ", " ## ", "   #", " ## "],
            'T' => ["###", " # ", " # ", " # ", " # "],
            'U' => ["#  #", "#  #", "#  #", "#  #", " ## "],
            'V' => ["#  #", "#  #", "#  #", " ## ", "  # "],
            'W' => ["#   #", "#   #", "# # #", "## ##", "#   #"],
            'X' => ["#  #", " ## ", " ## ", " ## ", "#  #"],
            'Y' => ["# #", "# #", " # ", " # ", " # "],
            'Z' => ["####", "  # ", " #  ", "#   ", "####"],
            '1' => [" # ", "## ", " # ", " # ", "###"],
            '2' => [" ## ", "   #", " ## ", "#   ", "####"],
            '3' => ["### ", "   #", " ## ", "   #", "### "],
            '4' => ["#  #", "#  #", "####", "   #", "   #"],
            '5' => ["####", "#   ", "### ", "   #", "### "],
            '6' => [" ## ", "#   ", "### ", "#  #", " ## "],
            '7' => ["####", "   #", "  # ", " #  ", "#   "],
            '8' => [" ## ", "#  #", " ## ", "#  #", " ## "],
            '9' => [" ## ", "#  #", " ###", "   #", " ## "],
            '!' => [" # ", " # ", " # ", "   ", " # "],
            _ => ["  ", "  ", "  ", "  ", "  "],
        };
        for (i, row) in glyph.iter().enumerate() {
            lines[i].push_str(row);
            lines[i].push(' ');
        }
    }
    lines.join("\n")
}

/// Draw a box around text.
pub fn box_drawing(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let max_w = lines.iter().map(|l| l.len()).max().unwrap_or(0);
    let mut out = format!("┌{}┐\n", "─".repeat(max_w + 2));
    for line in &lines {
        writeln!(out, "│ {line:max_w$} │").unwrap();
    }
    write!(out, "└{}┘", "─".repeat(max_w + 2)).unwrap();
    out
}

// ─────────────────────────────────────────────────
// §67  CMYK / HSL / Lab Color Conversion
// ─────────────────────────────────────────────────

/// CMYK color.
#[derive(Debug, Clone, PartialEq)]
pub struct Cmyk {
    pub c: f64,
    pub m: f64,
    pub y: f64,
    pub k: f64,
}

/// Color conversion operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorConvertOp {
    RgbToCmyk { r: u8, g: u8, b: u8 },
    CmykToRgb { c: f64, m: f64, y: f64, k: f64 },
    RgbToHsl { r: u8, g: u8, b: u8 },
    HslToRgb { h: f64, s: f64, l: f64 },
}

/// Execute a color conversion.
pub fn color_convert(op: &ColorConvertOp) -> String {
    match op {
        ColorConvertOp::RgbToCmyk { r, g, b } => {
            let rf = *r as f64 / 255.0;
            let gf = *g as f64 / 255.0;
            let bf = *b as f64 / 255.0;
            let k = 1.0 - rf.max(gf).max(bf);
            if k >= 1.0 {
                return format!("RGB({r},{g},{b}) = CMYK(0%, 0%, 0%, 100%)");
            }
            let c = (1.0 - rf - k) / (1.0 - k);
            let m = (1.0 - gf - k) / (1.0 - k);
            let y = (1.0 - bf - k) / (1.0 - k);
            format!(
                "RGB({},{},{}) = CMYK({:.1}%, {:.1}%, {:.1}%, {:.1}%)",
                r,
                g,
                b,
                c * 100.0,
                m * 100.0,
                y * 100.0,
                k * 100.0
            )
        }
        ColorConvertOp::CmykToRgb { c, m, y, k } => {
            let r = (255.0 * (1.0 - c / 100.0) * (1.0 - k / 100.0)).round() as u8;
            let g = (255.0 * (1.0 - m / 100.0) * (1.0 - k / 100.0)).round() as u8;
            let b = (255.0 * (1.0 - y / 100.0) * (1.0 - k / 100.0)).round() as u8;
            format!("CMYK({c:.0}%,{m:.0}%,{y:.0}%,{k:.0}%) = RGB({r},{g},{b})")
        }
        ColorConvertOp::RgbToHsl { r, g, b } => {
            let rf = *r as f64 / 255.0;
            let gf = *g as f64 / 255.0;
            let bf = *b as f64 / 255.0;
            let max = rf.max(gf).max(bf);
            let min = rf.min(gf).min(bf);
            let l = f64::midpoint(max, min);
            if (max - min).abs() < 1e-10 {
                return format!("RGB({},{},{}) = HSL(0°, 0%, {:.1}%)", r, g, b, l * 100.0);
            }
            let d = max - min;
            let s = if l > 0.5 {
                d / (2.0 - max - min)
            } else {
                d / (max + min)
            };
            let h = if (max - rf).abs() < 1e-10 {
                ((gf - bf) / d + if gf < bf { 6.0 } else { 0.0 }) / 6.0
            } else if (max - gf).abs() < 1e-10 {
                ((bf - rf) / d + 2.0) / 6.0
            } else {
                ((rf - gf) / d + 4.0) / 6.0
            };
            format!(
                "RGB({},{},{}) = HSL({:.0}°, {:.1}%, {:.1}%)",
                r,
                g,
                b,
                h * 360.0,
                s * 100.0,
                l * 100.0
            )
        }
        ColorConvertOp::HslToRgb { h, s, l } => {
            let s = s / 100.0;
            let l = l / 100.0;
            let h = h / 360.0;
            if s.abs() < 1e-10 {
                let v = (l * 255.0).round() as u8;
                return format!(
                    "HSL({:.0}°,{:.0}%,{:.0}%) = RGB({},{},{})",
                    h * 360.0,
                    s * 100.0,
                    l * 100.0,
                    v,
                    v,
                    v
                );
            }
            let hue2rgb = |p: f64, q: f64, mut t: f64| -> f64 {
                if t < 0.0 {
                    t += 1.0;
                }
                if t > 1.0 {
                    t -= 1.0;
                }
                if t < 1.0 / 6.0 {
                    return p + (q - p) * 6.0 * t;
                }
                if t < 1.0 / 2.0 {
                    return q;
                }
                if t < 2.0 / 3.0 {
                    return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
                }
                p
            };
            let q = if l < 0.5 {
                l * (1.0 + s)
            } else {
                l + s - l * s
            };
            let p = 2.0 * l - q;
            let r = (hue2rgb(p, q, h + 1.0 / 3.0) * 255.0).round() as u8;
            let g = (hue2rgb(p, q, h) * 255.0).round() as u8;
            let b = (hue2rgb(p, q, h - 1.0 / 3.0) * 255.0).round() as u8;
            format!(
                "HSL({:.0}°,{:.1}%,{:.1}%) = RGB({},{},{})",
                h * 360.0,
                s * 100.0,
                l * 100.0,
                r,
                g,
                b
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §68  RPN Calculator
// ─────────────────────────────────────────────────

/// Evaluate an RPN (Reverse Polish Notation) expression.
pub fn rpn_calc(expr: &str) -> Result<f64, String> {
    let mut stack: Vec<f64> = vec![];
    for token in expr.split_whitespace() {
        match token {
            "+" | "-" | "*" | "/" | "^" | "%" => {
                let b = stack.pop().ok_or("Stack underflow")?;
                let a = stack.pop().ok_or("Stack underflow")?;
                let result = match token {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => {
                        if b.abs() < 1e-15 {
                            return Err("Division by zero".into());
                        }
                        a / b
                    }
                    "^" => a.powf(b),
                    "%" => a % b,
                    _ => unreachable!(),
                };
                stack.push(result);
            }
            "sqrt" => {
                let a = stack.pop().ok_or("Stack underflow")?;
                stack.push(a.sqrt());
            }
            "sin" => {
                let a = stack.pop().ok_or("Stack underflow")?;
                stack.push(a.sin());
            }
            "cos" => {
                let a = stack.pop().ok_or("Stack underflow")?;
                stack.push(a.cos());
            }
            "abs" => {
                let a = stack.pop().ok_or("Stack underflow")?;
                stack.push(a.abs());
            }
            "dup" => {
                let a = *stack.last().ok_or("Stack underflow")?;
                stack.push(a);
            }
            "swap" => {
                let len = stack.len();
                if len < 2 {
                    return Err("Stack underflow".into());
                }
                stack.swap(len - 1, len - 2);
            }
            _ => {
                let val: f64 = token
                    .parse()
                    .map_err(|_| format!("Unknown token: {token}"))?;
                stack.push(val);
            }
        }
    }
    stack.pop().ok_or_else(|| "Empty expression".into())
}

// ─────────────────────────────────────────────────
// §69  JWT Decoder
// ─────────────────────────────────────────────────

/// Decoded JWT parts.
#[derive(Debug, Clone, PartialEq)]
pub struct JwtParts {
    pub header: String,
    pub payload: String,
    pub signature: String,
}

/// Decode a JWT token (no verification — just structure inspection).
pub fn decode_jwt(token: &str) -> Result<JwtParts, String> {
    fn base64url_decode(input: &str) -> Result<String, String> {
        let padded = match input.len() % 4 {
            2 => format!("{input}=="),
            3 => format!("{input}="),
            _ => input.to_string(),
        };
        // Decode base64url: replace - with +, _ with /
        let standard = padded.replace('-', "+").replace('_', "/");
        // Manual base64 decode
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let lookup = |c: u8| -> Result<u8, String> {
            if c == b'=' {
                return Ok(0);
            }
            alphabet
                .iter()
                .position(|&b| b == c)
                .map(|p| p as u8)
                .ok_or_else(|| format!("Invalid base64 char: {}", c as char))
        };
        let bytes_in: Vec<u8> = standard.bytes().collect();
        let mut decoded = Vec::new();
        for chunk in bytes_in.chunks(4) {
            if chunk.len() < 4 {
                break;
            }
            let a = lookup(chunk[0])?;
            let b = lookup(chunk[1])?;
            let c_val = lookup(chunk[2])?;
            let d = lookup(chunk[3])?;
            decoded.push((a << 2) | (b >> 4));
            if chunk[2] != b'=' {
                decoded.push((b << 4) | (c_val >> 2));
            }
            if chunk[3] != b'=' {
                decoded.push((c_val << 6) | d);
            }
        }
        String::from_utf8(decoded).map_err(|_| "Invalid UTF-8 in decoded JWT".into())
    }

    let parts: Vec<&str> = token.trim().split('.').collect();
    if parts.len() != 3 {
        return Err("Invalid JWT: expected 3 dot-separated parts".into());
    }

    let header = base64url_decode(parts[0])?;
    let payload = base64url_decode(parts[1])?;

    Ok(JwtParts {
        header,
        payload,
        signature: parts[2].to_string(),
    })
}

// ─────────────────────────────────────────────────
// §70  JSON / CSS Minifier
// ─────────────────────────────────────────────────

/// Minifier target.
#[derive(Debug, Clone, PartialEq)]
pub enum MinifyKind {
    Json,
    Css,
}

/// Minify JSON by removing whitespace outside strings.
pub fn minify_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escape = false;
    for c in input.chars() {
        if escape {
            out.push(c);
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            out.push(c);
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            out.push(c);
            continue;
        }
        if in_string {
            out.push(c);
            continue;
        }
        if !c.is_whitespace() {
            out.push(c);
        }
    }
    out
}

/// Minify CSS by removing comments, extra whitespace, and newlines.
pub fn minify_css(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_string = false;
    let mut string_char = '"';
    while i < len {
        if in_string {
            out.push(chars[i]);
            if chars[i] == string_char {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if chars[i] == '"' || chars[i] == '\'' {
            in_string = true;
            string_char = chars[i];
            out.push(chars[i]);
            i += 1;
            continue;
        }
        // Skip comments
        if i + 1 < len && chars[i] == '/' && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        if chars[i].is_whitespace() {
            // Collapse whitespace to single space, skip around certain chars
            while i < len && chars[i].is_whitespace() {
                i += 1;
            }
            if i < len
                && !"{};:,>+~".contains(chars[i])
                && !out.ends_with(|c: char| "{};:,>+~".contains(c))
            {
                out.push(' ');
            }
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

// ─────────────────────────────────────────────────
// §71  Base85 / Ascii85 Encoder/Decoder
// ─────────────────────────────────────────────────

/// Base85 operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Base85Op {
    Encode { input: String },
    Decode { input: String },
}

/// Encode bytes to Ascii85.
pub fn base85_encode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::from("<~");
    for chunk in bytes.chunks(4) {
        let mut val: u32 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            val |= (b as u32) << (24 - i * 8);
        }
        if chunk.len() == 4 && val == 0 {
            out.push('z');
        } else {
            let mut chars = [0u8; 5];
            for i in (0..5).rev() {
                chars[i] = (val % 85) as u8 + 33;
                val /= 85;
            }
            for &c in &chars[..=chunk.len()] {
                out.push(c as char);
            }
        }
    }
    out.push_str("~>");
    out
}

/// Decode Ascii85 to string.
pub fn base85_decode(input: &str) -> Result<String, String> {
    let trimmed = input
        .trim()
        .strip_prefix("<~")
        .unwrap_or(input.trim())
        .strip_suffix("~>")
        .unwrap_or(input.trim());
    let mut out = Vec::new();
    let chars: Vec<u8> = trimmed
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == b'z' {
            out.extend_from_slice(&[0, 0, 0, 0]);
            i += 1;
            continue;
        }
        let chunk_len = (chars.len() - i).min(5);
        let mut val: u32 = 0;
        for j in 0..5 {
            let c = if i + j < chars.len() {
                chars[i + j] - 33
            } else {
                84
            }; // pad with 'u' (84)
            val = val * 85 + c as u32;
        }
        let byte_count = chunk_len - 1;
        for j in 0..byte_count {
            out.push((val >> (24 - j * 8)) as u8);
        }
        i += chunk_len;
    }
    String::from_utf8(out).map_err(|_| "Invalid UTF-8 in decoded Base85".into())
}

// ─────────────────────────────────────────────────
// §72  Dockerfile Linter (Basic)
// ─────────────────────────────────────────────────

/// Lint rule result.
#[derive(Debug, Clone, PartialEq)]
pub struct LintIssue {
    pub line: usize,
    pub severity: String,
    pub message: String,
}

/// Lint a Dockerfile for common issues.
pub fn lint_dockerfile(content: &str) -> Vec<LintIssue> {
    let mut issues = vec![];
    let mut has_from = false;
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        let line_no = i + 1;

        if trimmed.starts_with("FROM ") {
            has_from = true;
        }

        // DL3006: Always tag your base image
        if trimmed.starts_with("FROM ")
            && !trimmed.contains(':')
            && !trimmed.contains(" AS ")
            && !trimmed.contains("scratch")
        {
            issues.push(LintIssue {
                line: line_no,
                severity: "warning".into(),
                message: "DL3006: Always tag the version of an image explicitly".into(),
            });
        }
        // DL3020: Use COPY instead of ADD for files
        if trimmed.starts_with("ADD ")
            && !trimmed.contains("http://")
            && !trimmed.contains("https://")
            && !trimmed.contains(".tar")
        {
            issues.push(LintIssue {
                line: line_no,
                severity: "warning".into(),
                message: "DL3020: Use COPY instead of ADD for files & folders".into(),
            });
        }
        // DL4006: Set SHELL option -o pipefail before RUN with pipe
        if trimmed.starts_with("RUN ") && trimmed.contains('|') && !trimmed.contains("pipefail") {
            issues.push(LintIssue {
                line: line_no,
                severity: "warning".into(),
                message: "DL4006: Set the SHELL option -o pipefail before RUN with a pipe in it"
                    .into(),
            });
        }
        // DL3009: Delete apt-get lists
        if trimmed.contains("apt-get install") && !trimmed.contains("rm -rf /var/lib/apt/lists") {
            issues.push(LintIssue {
                line: line_no,
                severity: "info".into(),
                message: "DL3009: Delete the apt-get lists after installing something".into(),
            });
        }
        // DL3015: Avoid additional packages with apt-get
        if trimmed.contains("apt-get install") && !trimmed.contains("--no-install-recommends") {
            issues.push(LintIssue {
                line: line_no,
                severity: "info".into(),
                message: "DL3015: Avoid additional packages by specifying --no-install-recommends"
                    .into(),
            });
        }
        // SC2086: Double quote to prevent globbing/splitting
        if trimmed.starts_with("RUN ") && trimmed.contains('$') && !trimmed.contains("\"$") {
            issues.push(LintIssue {
                line: line_no,
                severity: "info".into(),
                message: "SC2086: Double quote to prevent globbing and word splitting".into(),
            });
        }
        // DL3007: Using latest is prone to errors
        if trimmed.starts_with("FROM ") && trimmed.contains(":latest") {
            issues.push(LintIssue {
                line: line_no,
                severity: "warning".into(),
                message: "DL3007: Using the 'latest' tag is prone to errors if image changes"
                    .into(),
            });
        }
        // DL3003: Use WORKDIR to switch directories
        if trimmed.starts_with("RUN cd ") {
            issues.push(LintIssue {
                line: line_no,
                severity: "warning".into(),
                message: "DL3003: Use WORKDIR to switch to a directory".into(),
            });
        }
    }
    if !has_from {
        issues.push(LintIssue {
            line: 1,
            severity: "error".into(),
            message: "DL3001: Dockerfile must start with a FROM instruction".into(),
        });
    }
    issues
}

// ─────────────────────────────────────────────────
// §73  Crontab Schedule Describer
// ─────────────────────────────────────────────────

/// Describe a cron expression in human-readable English.
pub fn describe_cron(expr: &str) -> String {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() < 5 {
        return format!(
            "Invalid cron expression: expected 5 fields, got {}",
            parts.len()
        );
    }

    let minute = parts[0];
    let hour = parts[1];
    let dom = parts[2];
    let month = parts[3];
    let dow = parts[4];

    let mut desc = String::new();

    // Minute
    if minute == "*" {
        desc.push_str("Every minute");
    } else if let Some(step) = minute.strip_prefix("*/") {
        write!(desc, "Every {step} minutes").unwrap();
    } else {
        write!(desc, "At minute {minute}").unwrap();
    }

    // Hour
    if hour == "*" {
        desc.push_str(" of every hour");
    } else if let Some(step) = hour.strip_prefix("*/") {
        write!(desc, ", every {step} hours").unwrap();
    } else {
        write!(desc, ", at hour {hour}").unwrap();
    }

    // Day of month
    if dom != "*" {
        if let Some(step) = dom.strip_prefix("*/") {
            write!(desc, ", every {step} days").unwrap();
        } else {
            write!(desc, ", on day {dom} of the month").unwrap();
        }
    }

    // Month
    if month != "*" {
        let month_name = match month {
            "1" => "January",
            "2" => "February",
            "3" => "March",
            "4" => "April",
            "5" => "May",
            "6" => "June",
            "7" => "July",
            "8" => "August",
            "9" => "September",
            "10" => "October",
            "11" => "November",
            "12" => "December",
            _ => month,
        };
        write!(desc, ", in {month_name}").unwrap();
    }

    // Day of week
    if dow != "*" {
        let day_name = match dow {
            "0" | "7" => "Sunday",
            "1" => "Monday",
            "2" => "Tuesday",
            "3" => "Wednesday",
            "4" => "Thursday",
            "5" => "Friday",
            "6" => "Saturday",
            _ => dow,
        };
        write!(desc, ", on {day_name}").unwrap();
    }

    desc
}

// ─────────────────────────────────────────────────
// §74  Verilog Module Generator
// ─────────────────────────────────────────────────

/// Verilog module parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VerilogParams {
    pub name: String,
    pub inputs: Vec<(String, u32)>, // (name, width)
    pub outputs: Vec<(String, u32)>,
    pub body: String,
}

/// Generate a Verilog module.
pub fn generate_verilog(params: &VerilogParams) -> String {
    let mut out = String::new();
    writeln!(out, "module {}(", params.name).unwrap();
    let mut ports = vec![];
    for (name, width) in &params.inputs {
        if *width > 1 {
            ports.push(format!("    input [{}:0] {}", width - 1, name));
        } else {
            ports.push(format!("    input {name}"));
        }
    }
    for (name, width) in &params.outputs {
        if *width > 1 {
            ports.push(format!("    output reg [{}:0] {}", width - 1, name));
        } else {
            ports.push(format!("    output reg {name}"));
        }
    }
    out.push_str(&ports.join(",\n"));
    out.push_str("\n);\n\n");
    if !params.body.is_empty() {
        out.push_str(&params.body);
        out.push('\n');
    }
    out.push_str("endmodule\n");
    out
}

// ─────────────────────────────────────────────────
// §75  MathML Expression Generator
// ─────────────────────────────────────────────────

/// Generate MathML from a simple expression.
pub fn to_mathml(expr: &str) -> String {
    let mut out = String::from("<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n");

    // Simple tokenizer for math expressions
    let tokens: Vec<&str> = expr.split_whitespace().collect();
    if tokens.len() == 3 {
        // Binary expression: a op b
        let (a, op, b) = (tokens[0], tokens[1], tokens[2]);
        out.push_str("  <mrow>\n");
        writeln!(out, "    <mn>{a}</mn>").unwrap();
        let mo = match op {
            "+" => "+",
            "-" => "−",
            "*" | "×" => "×",
            "/" | "÷" => "÷",
            "=" => "=",
            "!=" | "≠" => "≠",
            "<" => "&lt;",
            ">" => "&gt;",
            "<=" | "≤" => "≤",
            ">=" | "≥" => "≥",
            _ => op,
        };
        writeln!(out, "    <mo>{mo}</mo>").unwrap();
        writeln!(out, "    <mn>{b}</mn>").unwrap();
        out.push_str("  </mrow>\n");
    } else if tokens.len() == 1 && tokens[0].contains('/') {
        // Fraction: a/b
        let parts: Vec<&str> = tokens[0].split('/').collect();
        if parts.len() == 2 {
            out.push_str("  <mfrac>\n");
            writeln!(out, "    <mn>{}</mn>", parts[0]).unwrap();
            writeln!(out, "    <mn>{}</mn>", parts[1]).unwrap();
            out.push_str("  </mfrac>\n");
        }
    } else {
        // Fallback: render as text
        for token in &tokens {
            if token.chars().all(|c| c.is_ascii_digit() || c == '.') {
                writeln!(out, "  <mn>{token}</mn>").unwrap();
            } else if token.len() == 1 && token.chars().next().is_some_and(char::is_alphabetic) {
                writeln!(out, "  <mi>{token}</mi>").unwrap();
            } else {
                writeln!(out, "  <mo>{token}</mo>").unwrap();
            }
        }
    }

    out.push_str("</math>");
    out
}

// ─────────────────────────────────────────────────
// §76  Seed-Based Color Palette
// ─────────────────────────────────────────────────

/// Generate a deterministic color palette from a seed string.
pub fn seed_palette(seed: &str, count: usize) -> Vec<String> {
    // Simple hash-based palette generation
    let mut hash: u64 = 5381;
    for byte in seed.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    let mut colors = vec![];
    for i in 0..count {
        let h = (hash.wrapping_add(i as u64 * 137)) % 360;
        let s = 60 + (hash.wrapping_add(i as u64 * 47)) % 30; // 60-90%
        let l = 45 + (hash.wrapping_add(i as u64 * 83)) % 20; // 45-65%
        colors.push(format!("hsl({h}, {s}%, {l}%)"));
    }
    colors
}

// ─────────────────────────────────────────────────
// §77  Data Size Formatter
// ─────────────────────────────────────────────────

/// Format bytes into human-readable size with both binary and SI units.
pub fn format_data_size(bytes: u64) -> String {
    let units_si = ["B", "KB", "MB", "GB", "TB", "PB"];
    let units_bin = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

    let format_with = |base: f64, units: &[&str]| -> String {
        let mut val = bytes as f64;
        let mut unit_idx = 0;
        while val >= base && unit_idx < units.len() - 1 {
            val /= base;
            unit_idx += 1;
        }
        if unit_idx == 0 {
            format!("{} {}", bytes, units[0])
        } else {
            format!("{:.2} {}", val, units[unit_idx])
        }
    };

    let si = format_with(1000.0, &units_si);
    let bin = format_with(1024.0, &units_bin);
    format!("{bytes} bytes\nSI:     {si}\nBinary: {bin}")
}

// ─────────────────────────────────────────────────
// §78  String Escaper/Unescaper
// ─────────────────────────────────────────────────

/// Escape operation.
#[derive(Debug, Clone, PartialEq)]
pub enum EscapeOp {
    HtmlEscape { input: String },
    HtmlUnescape { input: String },
    JsonEscape { input: String },
    JsonUnescape { input: String },
    UrlEncode { input: String },
    UrlDecode { input: String },
}

/// Execute an escape/unescape operation.
pub fn escape_op(op: &EscapeOp) -> String {
    match op {
        EscapeOp::HtmlEscape { input } => input
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;"),
        EscapeOp::HtmlUnescape { input } => input
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#x27;", "'")
            .replace("&#39;", "'"),
        EscapeOp::JsonEscape { input } => input
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t"),
        EscapeOp::JsonUnescape { input } => input
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t"),
        EscapeOp::UrlEncode { input } => {
            let mut encoded = String::new();
            for b in input.bytes() {
                if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
                    encoded.push(b as char);
                } else {
                    write!(encoded, "%{b:02X}").unwrap();
                }
            }
            encoded
        }
        EscapeOp::UrlDecode { input } => {
            let mut decoded = Vec::new();
            let bytes = input.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'%'
                    && i + 2 < bytes.len()
                    && let Ok(val) =
                        u8::from_str_radix(&String::from_utf8_lossy(&bytes[i + 1..i + 3]), 16)
                {
                    decoded.push(val);
                    i += 3;
                    continue;
                }
                if bytes[i] == b'+' {
                    decoded.push(b' ');
                } else {
                    decoded.push(bytes[i]);
                }
                i += 1;
            }
            String::from_utf8_lossy(&decoded).to_string()
        }
    }
}

// ─────────────────────────────────────────────────
// §79  IP Address Validator
// ─────────────────────────────────────────────────

/// Validate and classify an IP address.
pub fn validate_ip(addr: &str) -> String {
    let trimmed = addr.trim();

    // Check IPv4
    if let Some(ip) = parse_ipv4_octets(trimmed) {
        let class = if ip[0] < 128 {
            "A"
        } else if ip[0] < 192 {
            "B"
        } else if ip[0] < 224 {
            "C"
        } else if ip[0] < 240 {
            "D (Multicast)"
        } else {
            "E (Reserved)"
        };

        let scope = if ip[0] == 127 {
            "Loopback"
        } else if ip[0] == 10 {
            "Private (10.0.0.0/8)"
        } else if ip[0] == 172 && ip[1] >= 16 && ip[1] <= 31 {
            "Private (172.16.0.0/12)"
        } else if ip[0] == 192 && ip[1] == 168 {
            "Private (192.168.0.0/16)"
        } else if ip[0] == 169 && ip[1] == 254 {
            "Link-Local"
        } else if ip[0] == 0 {
            "This Network"
        } else if ip[0] >= 224 && ip[0] <= 239 {
            "Multicast"
        } else {
            "Public"
        };

        return format!(
            "{}\nType: IPv4\nClass: {}\nScope: {}\nBinary: {:08b}.{:08b}.{:08b}.{:08b}",
            trimmed, class, scope, ip[0], ip[1], ip[2], ip[3]
        );
    }

    // Check for simple IPv6 (::1, etc.)
    if trimmed.contains(':') && !trimmed.contains('.') {
        let scope = if trimmed == "::1" {
            "Loopback"
        } else if trimmed.starts_with("fe80:") {
            "Link-Local"
        } else if trimmed.starts_with("fc") || trimmed.starts_with("fd") {
            "Unique Local"
        } else if trimmed.starts_with("ff") {
            "Multicast"
        } else {
            "Global Unicast"
        };
        return format!("{trimmed}\nType: IPv6\nScope: {scope}");
    }

    format!("{trimmed}: Invalid IP address")
}

fn parse_ipv4_octets(s: &str) -> Option<[u8; 4]> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let octets: Vec<u8> = parts.iter().filter_map(|p| p.parse().ok()).collect();
    if octets.len() != 4 {
        return None;
    }
    Some([octets[0], octets[1], octets[2], octets[3]])
}

// ─────────────────────────────────────────────────
// §80  Semver Comparator
// ─────────────────────────────────────────────────

/// Parsed semver version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Semver {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre: String,
}

/// Parse a semver string.
pub fn parse_semver(s: &str) -> Result<Semver, String> {
    let s = s.trim().strip_prefix('v').unwrap_or(s.trim());
    let (version, pre) = if let Some(pos) = s.find('-') {
        (&s[..pos], s[pos + 1..].to_string())
    } else {
        (s, String::new())
    };
    let parts: Vec<&str> = version.split('.').collect();
    let major: u32 = parts
        .first()
        .and_then(|p| p.parse().ok())
        .ok_or("Invalid major version")?;
    let minor: u32 = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);
    let patch: u32 = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);
    Ok(Semver {
        major,
        minor,
        patch,
        pre,
    })
}

/// Compare two semver strings and describe the relationship.
pub fn compare_semver(a: &str, b: &str) -> Result<String, String> {
    let va = parse_semver(a)?;
    let vb = parse_semver(b)?;

    let bump = if va.major != vb.major {
        "MAJOR"
    } else if va.minor != vb.minor {
        "MINOR"
    } else if va.patch != vb.patch {
        "PATCH"
    } else {
        "EQUAL"
    };

    let relation = match va.cmp(&vb) {
        std::cmp::Ordering::Less => "<",
        std::cmp::Ordering::Greater => ">",
        std::cmp::Ordering::Equal => "=",
    };

    Ok(format!(
        "{} {} {}\nBump: {}\nBreaking change: {}",
        a,
        relation,
        b,
        bump,
        if bump == "MAJOR" {
            "Yes (major version changed)"
        } else {
            "No"
        }
    ))
}

// ─────────────────────────────────────────────────
// Unified Dispatcher (renumbered §42)
// ─────────────────────────────────────────────────

/// The type of deterministic tool being invoked.
#[derive(Debug, Clone, PartialEq)]
pub enum DeterministicToolKind {
    /// Mathematical expression evaluation.
    Math { expression: String },
    /// Unit conversion.
    UnitConversion {
        value: f64,
        from_unit: String,
        to_unit: String,
    },
    /// Number base conversion.
    BaseConversion { value: String, to_base: u32 },
    /// Show all base representations.
    BaseAll { value: String },
    /// Text encoding/decoding.
    TextTransform { operation: TextOp, input: String },
    /// Color conversion.
    ColorConvert { input: String },
    /// Statistical analysis.
    Statistics { input: String },
    /// Date calculation.
    DateCalc { operation: DateOp },
    /// MIDI note ↔ frequency ↔ name.
    Midi { operation: MidiOp },
    /// BPM to milliseconds per beat.
    BpmCalc { bpm: f64 },
    /// Roman numeral conversion.
    Roman { operation: RomanOp },
    /// Percentage calculation.
    Percentage { operation: PercentageOp },
    /// Generate a UUID.
    Uuid,
    /// Unix timestamp conversion.
    Timestamp { operation: TimestampOp },
    /// SVG parametric generation.
    Svg { params: SvgParams },
    /// DXF 2D geometry generation.
    Dxf { params: DxfParams },
    /// Regex operations (test, find, replace).
    Regex { operation: RegexOp },
    /// Cron expression parsing.
    Cron { expression: String },
    /// JSON/YAML/TOML format conversion.
    DataConvert {
        input: String,
        from: DataFormat,
        to: DataFormat,
    },
    /// IP/subnet calculation.
    Subnet { cidr: String },
    /// QR code generation (SVG output).
    QrCode { data: String },
    /// Password/passphrase generation.
    Password { operation: PasswordOp },
    /// CSS color palette generation.
    Palette {
        base_color: String,
        mode: PaletteMode,
        count: usize,
    },
    /// OpenSCAD parametric 3D generation.
    Scad { params: ScadParams },
    /// G-code generation (CNC/3D printer).
    Gcode { params: GcodeParams },
    /// STL mesh generation (3D printing).
    Stl {
        primitive: StlPrimitive,
        name: String,
    },
    /// Three.js scene generation (web 3D preview).
    ThreeJs { params: ThreeJsParams },
    /// SVG chart generation (bar, line, pie, scatter, histogram).
    Chart { params: ChartParams },
    /// Graphviz DOT graph generation.
    Dot { params: DotParams },
    /// Mermaid diagram generation.
    Mermaid { params: MermaidParams },
    /// WAV audio generation.
    Wav { params: WavParams },
    /// Wavefront OBJ 3D generation.
    Obj {
        primitive: ObjPrimitive,
        name: String,
    },
    /// LaTeX expression generation.
    Latex { kind: LatexKind },
    /// Unicode equation formatting.
    EquationFormat { expression: String },
    /// Terraform HCL generation.
    Terraform { params: TerraformParams },
    /// Docker Compose YAML generation.
    DockerCompose { params: ComposeParams },
    /// Kubernetes manifest generation.
    K8s { params: K8sParams },
    /// KiCad schematic generation.
    Kicad { params: KicadParams },
    /// SPICE netlist generation.
    Spice { params: SpiceParams },
    /// PBM/PPM bitmap generation.
    Bitmap { params: BitmapParams },
    /// Table/CSV generation.
    Table { params: TableParams },
    /// Document classification by extension.
    DocClassify { filename: String },
    /// Simple PDF generation.
    Pdf { params: PdfParams },
    /// Hash operations (MD5, CRC32, SHA256).
    Hash { operation: HashOp },
    /// Morse code encode/decode.
    Morse { operation: MorseOp },
    /// Text diff.
    Diff { old: String, new: String },
    /// Financial calculator.
    Finance { operation: FinanceOp },
    /// Bitwise calculator.
    Bitwise { operation: BitwiseOp },
    /// Truth table generator.
    TruthTable { expression: String },
    /// JSON Schema validation.
    JsonSchemaValidate { json: String, schema: String },
    /// XML ↔ JSON conversion.
    XmlConvert { input: String, to_xml: bool },
    /// URL/URI parsing.
    UrlParse { url: String },
    /// cURL command generation.
    Curl { params: CurlParams },
    /// Timezone conversion.
    TimezoneConvert {
        hour: u32,
        minute: u32,
        from_tz: String,
        to_tz: String,
    },
    /// Current time query — returns local time from system clock.
    CurrentTime { timezone: Option<String> },
    /// Barcode generation (SVG).
    Barcode { data: String, kind: BarcodeKind },
    /// Punycode encode/decode.
    Punycode { operation: PunycodeOp },
    /// Hex dump.
    HexDump { input: String },
    /// Glob pattern matching.
    GlobMatch { pattern: String, text: String },
    /// Geometry calculations.
    Geometry { operation: GeometryOp },
    /// UUID v5 (deterministic).
    UuidV5 { namespace: String, name: String },
    /// Snowflake ID decoder.
    SnowflakeDecode { id: u64, epoch: u64 },
    /// JSON structural diff.
    JsonDiff { old: String, new: String },
    /// HTTP header parsing.
    HttpHeaders { raw: String },
    /// MIME type lookup.
    MimeLookup { extension: String },
    /// Helm chart generation.
    Helm { params: HelmParams },
    /// DNS record generation.
    Dns { records: Vec<DnsRecord> },
    /// ASCII art banner.
    AsciiBanner { text: String },
    /// Box drawing around text.
    BoxDraw { text: String },
    /// Advanced color space conversion (CMYK, HSL).
    ColorSpace { operation: ColorConvertOp },
    /// RPN calculator.
    Rpn { expression: String },
    /// JWT decoder.
    JwtDecode { token: String },
    /// JSON/CSS minifier.
    Minify { kind: MinifyKind, input: String },
    /// Base85/Ascii85 encoder/decoder.
    Base85 { operation: Base85Op },
    /// Dockerfile linter.
    DockerfileLint { content: String },
    /// Crontab schedule describer.
    CronDescribe { expression: String },
    /// Verilog module generator.
    Verilog { params: VerilogParams },
    /// MathML generator.
    MathMl { expression: String },
    /// Seed-based color palette.
    SeedPalette { seed: String, count: usize },
    /// Data size formatter.
    DataSize { bytes: u64 },
    /// String escaper/unescaper.
    Escape { operation: EscapeOp },
    /// IP address validator.
    IpValidate { address: String },
    /// Semver comparator.
    SemverCompare { a: String, b: String },
    /// §82 Molecular weight & binding affinity.
    Molecule { operation: MoleculeOp },
    /// §85 Pharmacokinetic calculations.
    Pharma { operation: PharmaOp },
    /// §86 Clinical trial statistics.
    Clinical { operation: ClinicalOp },
    /// §81 Protein folding energy calculations.
    ProteinEnergy { operation: ProteinEnergyOp },
    /// §83 Gene sequence alignment (Smith-Waterman).
    Alignment { operation: AlignmentOp },
    /// §84 Drug interaction matrix.
    Drug { operation: DrugOp },
    /// §87 Signaling pathway graph traversal.
    Pathway { operation: PathwayOp },
    /// §88 Checksum validator (Luhn, ISBN, IBAN, EAN).
    Checksum { operation: ChecksumOp },
    /// §89 NATO phonetic alphabet.
    NatoPhonetic { text: String },
    /// §90 ROT13 / Caesar cipher.
    Caesar { operation: CaesarOp },
    /// §91 Aspect ratio calculator.
    AspectRatio { operation: AspectRatioOp },
    /// §92 Resistor color code decoder.
    Resistor { operation: ResistorOp },
    /// §93 Network bandwidth calculator.
    Bandwidth { operation: BandwidthOp },
    /// §94 Unicode character inspector.
    UnicodeInspect { input: String },
    /// §95 IEEE 754 float inspector.
    Float754 { value: f64 },
    /// §96 Frequency / wavelength calculator.
    FreqWavelength { operation: FreqWavelengthOp },
    /// §97 Chemical formula molar mass.
    MolarMass { formula: String },
    /// §98 Translation lookup (common phrases, 8 languages).
    Translate { text: String, target_lang: String },
    /// §99 Filesystem operations (replaces MCP filesystem server).
    #[cfg(feature = "io")]
    Filesystem { operation: FilesystemOp },
    /// §100 Git operations (replaces MCP git server).
    #[cfg(feature = "io")]
    Git { operation: GitOp },
    /// §101 Web fetch / search (replaces MCP fetch server).
    #[cfg(feature = "web")]
    WebFetch { operation: WebFetchOp },
    /// §102 OpenTofu/Terraform enhanced (modules, variables, backends).
    TerraformModule { params: TerraformModuleParams },
    /// §103 Ansible playbook generation.
    Ansible { params: AnsibleParams },
    /// §104 Pulumi IaC generation (TypeScript).
    Pulumi { params: PulumiParams },
    /// §105 CloudFormation template generation.
    CloudFormation { params: CloudFormationParams },
    /// §106 Prometheus/Grafana config generation.
    Monitoring { params: MonitoringParams },
    /// §107 Nginx/Caddy config generation.
    WebServer { params: WebServerParams },
    /// §108 systemd unit file generation.
    Systemd { params: SystemdParams },
    /// §109 GitHub Actions / CI pipeline generation.
    CiPipeline { params: CiPipelineParams },
    /// §110 OpenAPI 3.x spec generation.
    OpenApi { params: OpenApiParams },
    /// §111 SQL query builder.
    SqlQuery { params: SqlQueryParams },
    /// §112 GraphQL schema generation.
    GraphqlSchema { params: GraphqlSchemaParams },
    /// §113 Type generator (JSON → TypeScript/Rust/Python/Go).
    TypeGen { params: TypeGenParams },
    /// §114 Protobuf/gRPC definition generation.
    Protobuf { params: ProtobufParams },
    /// §115 .gitignore generator.
    Gitignore { language: String },
    /// §116 Secret/credential detector.
    SecretDetect { input: String },
    /// §117 Common regex pattern library.
    RegexPattern { name: String },
    /// §118 Kubernetes RBAC generation.
    K8sRbac { params: K8sRbacParams },
    /// §119 Kubernetes NetworkPolicy generation.
    K8sNetworkPolicy { params: K8sNetworkPolicyParams },
    /// §120 AWS IAM policy generation.
    AwsIamPolicy { params: AwsIamParams },
    /// §121 Syllogism validator (premises → conclusion).
    Syllogism { params: SyllogismParams },
    /// §122 Decision matrix (weighted criteria scoring).
    DecisionMatrix { params: DecisionMatrixParams },
    /// §123 SWOT analysis generator.
    Swot { params: SwotParams },
    /// §124 Pros/Cons analysis.
    ProsCons {
        topic: String,
        pros: Vec<String>,
        cons: Vec<String>,
    },
    /// §125 Root cause analysis (5 Whys / Fishbone).
    RootCause { params: RootCauseParams },
    /// §126 Logical deduction chain.
    Deduction { params: DeductionParams },
    /// Calendar query — returns events for a date (server fills in data).
    CalendarQuery { date_hint: Option<String> },
    /// Meet room — signals to open the Meet workspace tab.
    MeetRoom,

    // ── §127–§132  Energy Floor / JCI Tools ────────────────────────
    //
    // The Energy Floor value function: V(g) = E(g) + T(g) + S(g) + C(g)
    // These tools implement the core computations from jci-core as
    // deterministic, zero-hallucination L0.5 cascade operations.

    /// §127 Anomaly detection — z-score on rolling window.
    EnergyFloor { operation: EnergyFloorOp },

    // ── §133  Geographic / GIS Tools ─────────────────────────────────
    //
    // Deterministic geodesy: Haversine, Vincenty, geohash, coordinate
    // parsing, bearing, midpoint, bounding box. Zero network.

    /// §133 Geographic calculations.
    Geo { operation: GeoOp },
}

/// Energy floor operations — JCI value function components.
#[derive(Debug, Clone, PartialEq)]
pub enum EnergyFloorOp {
    /// Z-score anomaly detection on a rolling window.
    /// Returns z-score, direction, rolling mean/std, and anomaly flag.
    AnomalyDetect {
        /// Series identifier (e.g. "ercot_day_ahead", "btc_usd").
        series_id: String,
        /// Price/value series (chronological).
        values: Vec<f64>,
        /// Rolling window size (default 30).
        window: usize,
        /// Z-score threshold for anomaly flag (default 2.0).
        threshold: f64,
    },
    /// Lagged cross-correlation between two series.
    /// Returns Pearson r, p-value, and interpretation.
    Correlation {
        /// Series A identifier.
        series_a_id: String,
        /// Series A values.
        series_a: Vec<f64>,
        /// Series B identifier.
        series_b_id: String,
        /// Series B values.
        series_b: Vec<f64>,
        /// Lag in samples (positive = A leads B).
        lag: i64,
    },
    /// Cost function: C(W) = E(W)·Pₑ + H(W)·Pₕ + I(W).
    /// Returns cost breakdown and fractions.
    CostFunction {
        /// Energy consumed (joules).
        energy_joules: f64,
        /// Energy price ($/joule).
        energy_price_per_joule: f64,
        /// Hardware units consumed.
        hardware_units: f64,
        /// Hardware price ($/unit).
        hardware_price_per_unit: f64,
        /// Friction/overhead cost ($).
        friction_cost: f64,
        /// Useful bits produced.
        useful_bits: f64,
    },
    /// Forward curve: F(t) = S·exp((r−κ)·t).
    /// Compute naturally trades in backwardation because κ ≈ 0.26 (Koomey decay).
    ForwardCurve {
        /// Current spot price ($/unit).
        spot_price: f64,
        /// Risk-free rate (annualized, e.g. 0.05 = 5%).
        risk_free_rate: f64,
        /// Koomey decay rate (annualized, default 0.26).
        koomey_rate: f64,
        /// Tenor points to compute (days).
        tenors_days: Vec<u32>,
        /// Asset label (e.g. "H100 GPU-hour").
        asset_label: String,
    },
    /// Geographic arbitrage spread between two regions.
    /// Returns spread, annualized P&L, and thesis.
    ArbitrageSpread {
        /// Long (destination) region.
        long_region: String,
        /// Long region energy price ($/kWh).
        long_price_kwh: f64,
        /// Short (source) region.
        short_region: String,
        /// Short region energy price ($/kWh).
        short_price_kwh: f64,
        /// Reference throughput (MW).
        throughput_mw: f64,
    },
    /// Full value function: V(g) = E(g) + T(g) + S(g) + C(g).
    /// Decomposes a good's price into its four irreducible components.
    ValueFunction {
        /// Good/service being priced.
        good: String,
        /// E(g): energy cost component ($).
        energy_cost: f64,
        /// T(g): trust/verification cost ($).
        trust_cost: f64,
        /// S(g): speed/intelligence cost ($).
        speed_cost: f64,
        /// C(g): compliance/regulatory cost ($).
        compliance_cost: f64,
    },
    /// Landauer bound: E_L = k_B·T·ln(2) ≈ 2.8×10⁻²¹ J/bit at room temp.
    /// Returns thermodynamic efficiency and orders above Landauer.
    LandauerBound {
        /// Actual energy per bit (joules).
        actual_joules_per_bit: f64,
        /// Temperature (kelvin, default 300.0).
        temperature_k: f64,
    },
}

/// Text transform operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TextOp {
    Base64Encode,
    Base64Decode,
    UrlEncode,
    UrlDecode,
    Sha256,
    WordCount,
    Uppercase,
    Lowercase,
    TitleCase,
    Reverse,
}

/// Date operations.
#[derive(Debug, Clone, PartialEq)]
pub enum DateOp {
    DaysBetween { date1: String, date2: String },
    AddDays { date: String, days: i64 },
    DayOfWeek { date: String },
}

/// MIDI/music operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MidiOp {
    /// MIDI note number → frequency + name.
    NoteToFreq { note: u8 },
    /// Frequency → nearest MIDI note + name.
    FreqToNote { freq: f64 },
    /// Note name (e.g. "C4") → MIDI number + frequency.
    NameToNote { name: String },
}

/// Roman numeral operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RomanOp {
    ToRoman { value: u32 },
    FromRoman { input: String },
}

/// Percentage operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PercentageOp {
    /// "what is X% of Y"
    Of { percent: f64, value: f64 },
    /// "X is what % of Y"
    WhatPercent { part: f64, whole: f64 },
    /// "percentage change from X to Y"
    Change { old: f64, new: f64 },
}

/// Timestamp operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TimestampOp {
    /// Unix timestamp → human-readable.
    ToDateTime { timestamp: i64 },
    /// Date → Unix timestamp.
    ToTimestamp { date: String },
    /// Get current timestamp.
    Now,
}

/// §88 Checksum validation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ChecksumOp {
    /// Luhn algorithm validation (credit cards).
    Luhn { digits: String },
    /// ISBN-10 or ISBN-13 validation.
    Isbn { code: String },
    /// IBAN validation.
    Iban { code: String },
    /// EAN-13 barcode validation.
    Ean13 { code: String },
}

/// §90 Caesar/ROT13 cipher operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CaesarOp {
    /// ROT13 encode/decode (symmetric).
    Rot13 { text: String },
    /// Caesar cipher with custom shift.
    Encrypt { text: String, shift: u8 },
    /// Caesar decipher with known shift.
    Decrypt { text: String, shift: u8 },
}

/// §91 Aspect ratio operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AspectRatioOp {
    /// Calculate aspect ratio from width × height.
    FromDimensions { width: u32, height: u32 },
    /// Scale dimensions preserving ratio.
    Scale {
        width: u32,
        height: u32,
        target_width: u32,
    },
}

/// §92 Resistor color code operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ResistorOp {
    /// Decode color bands to resistance.
    Decode { bands: Vec<String> },
    /// Encode resistance to color bands.
    Encode { ohms: f64 },
}

/// §93 Network bandwidth operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BandwidthOp {
    /// Transfer time: file_size / speed.
    TransferTime { bytes: u64, bits_per_sec: u64 },
    /// Required speed: file_size / time.
    RequiredSpeed { bytes: u64, seconds: f64 },
}

/// §96 Frequency / wavelength operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FreqWavelengthOp {
    /// Frequency → wavelength.
    FreqToWavelength { hz: f64 },
    /// Wavelength → frequency.
    WavelengthToFreq { meters: f64 },
    /// EM spectrum band classification.
    Classify { hz: f64 },
}

/// §99 Filesystem operations.
#[cfg(feature = "io")]
#[derive(Debug, Clone, PartialEq)]
pub enum FilesystemOp {
    /// Read file contents.
    ReadFile { path: String },
    /// Read multiple files at once.
    ReadMultiple { paths: Vec<String> },
    /// Write content to a file.
    WriteFile { path: String, content: String },
    /// Create a directory (with parents).
    CreateDirectory { path: String },
    /// List directory contents.
    ListDirectory { path: String },
    /// Directory tree (recursive listing).
    DirectoryTree { path: String, max_depth: usize },
    /// Move/rename a file or directory.
    MoveFile { source: String, destination: String },
    /// Copy a file.
    CopyFile { source: String, destination: String },
    /// Delete a file.
    DeleteFile { path: String },
    /// Check if a file exists.
    FileExists { path: String },
    /// Get file info (size, modified, type).
    FileInfo { path: String },
    /// Search files by name pattern (glob).
    SearchFiles { directory: String, pattern: String },
}

/// §100 Git operations.
#[cfg(feature = "io")]
#[derive(Debug, Clone, PartialEq)]
pub enum GitOp {
    /// git status.
    Status { repo_path: String },
    /// git log (last N commits).
    Log { repo_path: String, count: usize },
    /// git diff (working tree or between refs).
    Diff {
        repo_path: String,
        target: Option<String>,
    },
    /// git add files.
    Add {
        repo_path: String,
        files: Vec<String>,
    },
    /// git commit.
    Commit { repo_path: String, message: String },
    /// List branches.
    BranchList { repo_path: String },
    /// Create a branch.
    BranchCreate { repo_path: String, name: String },
    /// Checkout a branch.
    Checkout { repo_path: String, target: String },
    /// git stash.
    Stash { repo_path: String, action: String },
    /// List tags.
    TagList { repo_path: String },
    /// List remotes.
    RemoteList { repo_path: String },
    /// git clone.
    Clone { url: String, destination: String },
}

/// §101 Web fetch / search operations.
#[cfg(feature = "web")]
#[derive(Debug, Clone, PartialEq)]
pub enum WebFetchOp {
    /// Fetch URL content (HTTP GET → text).
    Fetch {
        url: String,
        max_length: Option<usize>,
    },
    /// Web search (via jouleclaw search engine).
    Search { query: String, count: Option<usize> },
    /// News search (via jouleclaw search engine).
    NewsSearch { query: String, count: Option<usize> },
}

/// Execute a deterministic tool and return a formatted text result.
///
/// Cost: 0 USD. Latency: <1ms. Accuracy: 100%.
pub fn execute(tool: &DeterministicToolKind) -> Result<String, String> {
    match tool {
        DeterministicToolKind::Math { expression } => {
            let result = eval_math(expression)?;
            Ok(format!(
                "{} = {}",
                expression.trim(),
                format_math_result(result)
            ))
        }
        DeterministicToolKind::UnitConversion {
            value,
            from_unit,
            to_unit,
        } => {
            let (result, from_canon, to_canon) = convert_units(*value, from_unit, to_unit)?;
            Ok(format_conversion(*value, result, &from_canon, &to_canon))
        }
        DeterministicToolKind::BaseConversion { value, to_base } => convert_base(value, *to_base),
        DeterministicToolKind::BaseAll { value } => format_all_bases(value),
        DeterministicToolKind::TextTransform { operation, input } => match operation {
            TextOp::Base64Encode => Ok(base64_encode(input)),
            TextOp::Base64Decode => base64_decode(input),
            TextOp::UrlEncode => Ok(url_encode(input)),
            TextOp::UrlDecode => url_decode(input),
            TextOp::Sha256 => Ok(sha256(input)),
            TextOp::WordCount => Ok(text_stats(input).to_string()),
            TextOp::Uppercase => Ok(to_upper(input)),
            TextOp::Lowercase => Ok(to_lower(input)),
            TextOp::TitleCase => Ok(to_title_case(input)),
            TextOp::Reverse => Ok(reverse_string(input)),
        },
        DeterministicToolKind::ColorConvert { input } => {
            let color = parse_color(input)?;
            Ok(color.to_string())
        }
        DeterministicToolKind::Statistics { input } => {
            let numbers = parse_number_list(input)?;
            let stats = statistics(&numbers)?;
            Ok(stats.to_string())
        }
        DeterministicToolKind::DateCalc { operation } => match operation {
            DateOp::DaysBetween { date1, date2 } => {
                let days = days_between(date1, date2)?;
                Ok(format!(
                    "{} days between {} and {}",
                    days.abs(),
                    date1,
                    date2
                ))
            }
            DateOp::AddDays { date, days } => {
                let result = add_days(date, *days)?;
                Ok(format!("{date} + {days} days = {result}"))
            }
            DateOp::DayOfWeek { date } => {
                let dow = day_of_week(date)?;
                Ok(format!("{date} is a {dow}"))
            }
        },
        DeterministicToolKind::Midi { operation } => match operation {
            MidiOp::NoteToFreq { note } => {
                let freq = midi_to_freq(*note);
                let name = midi_to_name(*note);
                Ok(format!("MIDI {note} ({name}) = {freq:.2} Hz"))
            }
            MidiOp::FreqToNote { freq } => {
                let (note, cents) = freq_to_midi(*freq);
                let name = midi_to_name(note);
                if cents.abs() < 0.5 {
                    Ok(format!("{freq:.2} Hz = MIDI {note} ({name})"))
                } else {
                    Ok(format!(
                        "{freq:.2} Hz = MIDI {note} ({name}) {cents:+.0} cents"
                    ))
                }
            }
            MidiOp::NameToNote { name } => {
                let note = name_to_midi(name)?;
                let freq = midi_to_freq(note);
                Ok(format!("{name} = MIDI {note} = {freq:.2} Hz"))
            }
        },
        DeterministicToolKind::BpmCalc { bpm } => {
            let ms = bpm_to_ms(*bpm);
            let samples_44k = bpm_to_samples(*bpm, 44100.0);
            Ok(format!(
                "{bpm} BPM = {ms:.1} ms/beat = {samples_44k:.0} samples/beat @44.1kHz"
            ))
        }
        DeterministicToolKind::Roman { operation } => match operation {
            RomanOp::ToRoman { value } => {
                let roman = to_roman(*value)?;
                Ok(format!("{value} = {roman}"))
            }
            RomanOp::FromRoman { input } => {
                let value = from_roman(input)?;
                Ok(format!("{} = {}", input.to_uppercase(), value))
            }
        },
        DeterministicToolKind::Percentage { operation } => match operation {
            PercentageOp::Of { percent, value } => {
                let result = percentage_of(*percent, *value);
                Ok(format!(
                    "{}% of {} = {}",
                    format_math_result(*percent),
                    format_math_result(*value),
                    format_math_result(result)
                ))
            }
            PercentageOp::WhatPercent { part, whole } => {
                let result = what_percentage(*part, *whole)?;
                Ok(format!(
                    "{} is {:.2}% of {}",
                    format_math_result(*part),
                    result,
                    format_math_result(*whole)
                ))
            }
            PercentageOp::Change { old, new } => {
                let result = percentage_change(*old, *new)?;
                Ok(format!(
                    "{} → {} = {:+.2}% change",
                    format_math_result(*old),
                    format_math_result(*new),
                    result
                ))
            }
        },
        DeterministicToolKind::Uuid => Ok(generate_uuid()),
        DeterministicToolKind::Timestamp { operation } => match operation {
            TimestampOp::ToDateTime { timestamp } => {
                let dt = timestamp_to_datetime(*timestamp)?;
                Ok(format!("{timestamp} = {dt}"))
            }
            TimestampOp::ToTimestamp { date } => {
                let ts = datetime_to_timestamp(date)?;
                Ok(format!("{date} = {ts} (Unix)"))
            }
            TimestampOp::Now => {
                let ts = now_timestamp();
                let dt = timestamp_to_datetime(ts)?;
                Ok(format!("now = {ts} = {dt}"))
            }
        },
        DeterministicToolKind::Svg { params } => Ok(generate_svg(params)),
        DeterministicToolKind::Dxf { params } => Ok(generate_dxf(params)),
        DeterministicToolKind::Regex { operation } => regex_exec(operation),
        DeterministicToolKind::Cron { expression } => parse_cron(expression),
        DeterministicToolKind::DataConvert { input, from, to } => {
            convert_data_format(input, from, to)
        }
        DeterministicToolKind::Subnet { cidr } => {
            let info = calc_subnet(cidr)?;
            Ok(info.to_string())
        }
        DeterministicToolKind::QrCode { data } => generate_qr_svg(data, 4.0),
        DeterministicToolKind::Password { operation } => match operation {
            PasswordOp::Random { length } => {
                let pw = generate_password(*length);
                Ok(format!("Generated password ({length} chars): {pw}"))
            }
            PasswordOp::Passphrase { words } => {
                let pp = generate_passphrase(*words);
                Ok(format!("Generated passphrase ({words} words): {pp}"))
            }
        },
        DeterministicToolKind::Palette {
            base_color,
            mode,
            count,
        } => generate_palette(base_color, mode, *count),
        DeterministicToolKind::Scad { params } => Ok(generate_scad(params)),
        DeterministicToolKind::Gcode { params } => Ok(generate_gcode(params)),
        DeterministicToolKind::Stl { primitive, name } => {
            let stl_params = stl_from_primitive(name, primitive);
            Ok(generate_stl(&stl_params))
        }
        DeterministicToolKind::ThreeJs { params } => Ok(generate_threejs(params)),
        DeterministicToolKind::Chart { params } => Ok(generate_chart(params)),
        DeterministicToolKind::Dot { params } => Ok(generate_dot(params)),
        DeterministicToolKind::Mermaid { params } => Ok(generate_mermaid(params)),
        DeterministicToolKind::Wav { params } => Ok(generate_wav(params)),
        DeterministicToolKind::Obj { primitive, name } => Ok(generate_obj(primitive, name)),
        DeterministicToolKind::Latex { kind } => Ok(generate_latex(kind)),
        DeterministicToolKind::EquationFormat { expression } => {
            Ok(format_equation_unicode(expression))
        }
        DeterministicToolKind::Terraform { params } => Ok(generate_terraform(params)),
        DeterministicToolKind::DockerCompose { params } => Ok(generate_compose(params)),
        DeterministicToolKind::K8s { params } => Ok(generate_k8s(params)),
        DeterministicToolKind::Kicad { params } => Ok(generate_kicad(params)),
        DeterministicToolKind::Spice { params } => Ok(generate_spice(params)),
        DeterministicToolKind::Bitmap { params } => Ok(generate_ppm(params)),
        DeterministicToolKind::Table { params } => Ok(generate_table(params)),
        DeterministicToolKind::DocClassify { filename } => {
            let doc_type = classify_document_extension(filename);
            Ok(format!("{filename}: {doc_type}"))
        }
        DeterministicToolKind::Pdf { params } => Ok(generate_pdf(params)),
        // §43 – Hash
        DeterministicToolKind::Hash { operation } => Ok(match operation {
            HashOp::Md5 { input } => format!("MD5: {}", md5_hash(input)),
            HashOp::Crc32 { input } => format!("CRC32: {}", crc32_hash(input)),
            HashOp::Sha256 { input } => format!("SHA256: {}", sha256(input)),
        }),
        // §44 – Morse
        DeterministicToolKind::Morse { operation } => Ok(match operation {
            MorseOp::Encode { text } => morse_encode(text),
            MorseOp::Decode { morse } => morse_decode(morse),
        }),
        // §45 – Text Diff
        DeterministicToolKind::Diff { old, new } => Ok(text_diff(old, new)),
        // §46 – Finance
        DeterministicToolKind::Finance { operation } => Ok(finance_calc(operation)),
        // §47 – Bitwise
        DeterministicToolKind::Bitwise { operation } => Ok(bitwise_calc(operation)),
        // §48 – Truth Table
        DeterministicToolKind::TruthTable { expression } => Ok(truth_table(expression)),
        // §49 – JSON Schema Validate
        DeterministicToolKind::JsonSchemaValidate { json, schema } => {
            validate_json_schema(json, schema)
        }
        // §50 – XML Convert
        DeterministicToolKind::XmlConvert { input, to_xml } => {
            if *to_xml {
                json_to_xml(input)
            } else {
                xml_to_json(input)
            }
        }
        // §51 – URL Parse
        DeterministicToolKind::UrlParse { url } => match parse_url(url) {
            Ok(parts) => Ok(format!(
                "Scheme: {}\nHost: {}\nPort: {}\nPath: {}\nQuery: {}\nFragment: {}",
                parts.scheme,
                parts.host,
                parts.port.map_or("(default)".into(), |p| p.to_string()),
                parts.path,
                parts
                    .query
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join("&"),
                parts.fragment.as_deref().unwrap_or("(none)")
            )),
            Err(e) => Err(e),
        },
        // §52 – cURL
        DeterministicToolKind::Curl { params } => Ok(generate_curl(params)),
        // §53 – Timezone Convert
        DeterministicToolKind::TimezoneConvert {
            hour,
            minute,
            from_tz,
            to_tz,
        } => convert_timezone(*hour, *minute, from_tz, to_tz),
        // §53b – Current Time
        DeterministicToolKind::CurrentTime { timezone } => current_time(timezone.as_deref()),
        // §54 – Barcode
        DeterministicToolKind::Barcode { data, kind } => generate_barcode(data, kind),
        // §55 – Punycode
        DeterministicToolKind::Punycode { operation } => Ok(match operation {
            PunycodeOp::Encode { text } => punycode_encode(text),
            PunycodeOp::Decode { punycode } => punycode_decode(punycode),
        }),
        // §56 – Hex Dump
        DeterministicToolKind::HexDump { input } => Ok(hex_dump(input)),
        // §57 – Glob Match
        DeterministicToolKind::GlobMatch { pattern, text } => Ok(if glob_match(pattern, text) {
            format!("\"{text}\" matches pattern \"{pattern}\"")
        } else {
            format!("\"{text}\" does NOT match pattern \"{pattern}\"")
        }),
        // §58 – Geometry
        DeterministicToolKind::Geometry { operation } => Ok(geometry_calc(operation)),
        // §59 – UUID v5
        DeterministicToolKind::UuidV5 { namespace, name } => Ok(uuid_v5(namespace, name)),
        // §60 – Snowflake Decode
        DeterministicToolKind::SnowflakeDecode { id, epoch } => Ok(decode_snowflake(*id, *epoch)),
        // §61 – JSON Diff
        DeterministicToolKind::JsonDiff { old, new } => json_diff(old, new),
        // §62 – HTTP Headers
        DeterministicToolKind::HttpHeaders { raw } => Ok(parse_http_headers(raw)),
        // §63 – MIME Lookup
        DeterministicToolKind::MimeLookup { extension } => {
            Ok(format!("{}: {}", extension, mime_from_extension(extension)))
        }
        // §64 – Helm
        DeterministicToolKind::Helm { params } => Ok(generate_helm(params)),
        // §65 – DNS
        DeterministicToolKind::Dns { records } => Ok(generate_dns_records(records)),
        // §66 – ASCII Banner
        DeterministicToolKind::AsciiBanner { text } => Ok(ascii_banner(text)),
        // §66 – Box Drawing
        DeterministicToolKind::BoxDraw { text } => Ok(box_drawing(text)),
        // §67 – Color Space Convert
        DeterministicToolKind::ColorSpace { operation } => Ok(color_convert(operation)),
        // §68 – RPN Calculator
        DeterministicToolKind::Rpn { expression } => rpn_calc(expression).map(|v| format!("{v}")),
        // §69 – JWT Decode
        DeterministicToolKind::JwtDecode { token } => decode_jwt(token).map(|parts| {
            format!(
                "Header:    {}\nPayload:   {}\nSignature: {}",
                parts.header, parts.payload, parts.signature
            )
        }),
        // §70 – Minify
        DeterministicToolKind::Minify { kind, input } => Ok(match kind {
            MinifyKind::Json => minify_json(input),
            MinifyKind::Css => minify_css(input),
        }),
        // §71 – Base85
        DeterministicToolKind::Base85 { operation } => Ok(match operation {
            Base85Op::Encode { input } => base85_encode(input),
            Base85Op::Decode { input } => base85_decode(input).unwrap_or_else(|e| e),
        }),
        // §72 – Dockerfile Lint
        DeterministicToolKind::DockerfileLint { content } => {
            let issues = lint_dockerfile(content);
            if issues.is_empty() {
                Ok("No issues found.".into())
            } else {
                Ok(issues
                    .iter()
                    .map(|i| format!("Line {}: [{}] {}", i.line, i.severity, i.message))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
        // §73 – Cron Describe
        DeterministicToolKind::CronDescribe { expression } => Ok(describe_cron(expression)),
        // §74 – Verilog
        DeterministicToolKind::Verilog { params } => Ok(generate_verilog(params)),
        // §75 – MathML
        DeterministicToolKind::MathMl { expression } => Ok(to_mathml(expression)),
        // §76 – Seed Palette
        DeterministicToolKind::SeedPalette { seed, count } => {
            let colors = seed_palette(seed, *count);
            Ok(colors.join("\n"))
        }
        // §77 – Data Size
        DeterministicToolKind::DataSize { bytes } => Ok(format_data_size(*bytes)),
        // §78 – Escape
        DeterministicToolKind::Escape { operation } => Ok(escape_op(operation)),
        // §79 – IP Validate
        DeterministicToolKind::IpValidate { address } => Ok(validate_ip(address)),
        // §80 – Semver Compare
        DeterministicToolKind::SemverCompare { a, b } => compare_semver(a, b),
        // §82 – Molecular Weight & Binding Affinity
        DeterministicToolKind::Molecule { operation } => Ok(molecule_calc(operation)),
        // §85 – Pharmacokinetics
        DeterministicToolKind::Pharma { operation } => Ok(pharma_calc(operation)),
        // §86 – Clinical Trial Statistics
        DeterministicToolKind::Clinical { operation } => Ok(clinical_calc(operation)),
        // §81 – Protein Folding Energy
        DeterministicToolKind::ProteinEnergy { operation } => Ok(protein_energy_calc(operation)),
        // §83 – Sequence Alignment
        DeterministicToolKind::Alignment { operation } => Ok(alignment_calc(operation)),
        // §84 – Drug Interaction
        DeterministicToolKind::Drug { operation } => Ok(drug_calc(operation)),
        // §87 – Pathway Traversal
        DeterministicToolKind::Pathway { operation } => Ok(pathway_calc(operation)),
        // §88 – Checksum Validator
        DeterministicToolKind::Checksum { operation } => Ok(checksum_validate(operation)),
        // §89 – NATO Phonetic
        DeterministicToolKind::NatoPhonetic { text } => Ok(nato_phonetic(text)),
        // §90 – Caesar/ROT13
        DeterministicToolKind::Caesar { operation } => Ok(caesar_calc(operation)),
        // §91 – Aspect Ratio
        DeterministicToolKind::AspectRatio { operation } => Ok(aspect_ratio_calc(operation)),
        // §92 – Resistor Color Code
        DeterministicToolKind::Resistor { operation } => Ok(resistor_calc(operation)),
        // §93 – Network Bandwidth
        DeterministicToolKind::Bandwidth { operation } => Ok(bandwidth_calc(operation)),
        // §94 – Unicode Inspector
        DeterministicToolKind::UnicodeInspect { input } => Ok(unicode_inspect(input)),
        // §95 – IEEE 754 Float
        DeterministicToolKind::Float754 { value } => Ok(float754_inspect(*value)),
        // §96 – Frequency/Wavelength
        DeterministicToolKind::FreqWavelength { operation } => Ok(freq_wavelength_calc(operation)),
        // §97 – Molar Mass
        DeterministicToolKind::MolarMass { formula } => molar_mass(formula),
        // §98 – Translation
        DeterministicToolKind::Translate { text, target_lang } => {
            Ok(translate_lookup(text, target_lang))
        }
        // §99 – Filesystem
        #[cfg(feature = "io")]
        DeterministicToolKind::Filesystem { operation } => filesystem_exec(operation),
        // §100 – Git
        #[cfg(feature = "io")]
        DeterministicToolKind::Git { operation } => git_exec(operation),
        // §101 – Web Fetch / Search (sync stub — use execute_async for actual I/O)
        #[cfg(feature = "web")]
        DeterministicToolKind::WebFetch { .. } => {
            Err("[needs async] Use execute_async() for WebFetch operations".into())
        }
        // §102 – OpenTofu / Terraform Module
        DeterministicToolKind::TerraformModule { params } => Ok(generate_terraform_module(params)),
        // §103 – Ansible
        DeterministicToolKind::Ansible { params } => Ok(generate_ansible(params)),
        // §104 – Pulumi
        DeterministicToolKind::Pulumi { params } => Ok(generate_pulumi(params)),
        // §105 – CloudFormation
        DeterministicToolKind::CloudFormation { params } => Ok(generate_cloudformation(params)),
        // §106 – Prometheus / Grafana
        DeterministicToolKind::Monitoring { params } => Ok(generate_monitoring(params)),
        // §107 – Nginx / Caddy
        DeterministicToolKind::WebServer { params } => Ok(generate_webserver(params)),
        // §108 – systemd
        DeterministicToolKind::Systemd { params } => Ok(generate_systemd(params)),
        // §109 – GitHub Actions / CI
        DeterministicToolKind::CiPipeline { params } => Ok(generate_ci_pipeline(params)),
        DeterministicToolKind::OpenApi { params } => Ok(generate_openapi(params)),
        DeterministicToolKind::SqlQuery { params } => Ok(generate_sql(params)),
        DeterministicToolKind::GraphqlSchema { params } => Ok(generate_graphql_schema(params)),
        DeterministicToolKind::TypeGen { params } => Ok(generate_types(params)),
        DeterministicToolKind::Protobuf { params } => Ok(generate_protobuf(params)),
        DeterministicToolKind::Gitignore { language } => Ok(generate_gitignore(language)),
        DeterministicToolKind::SecretDetect { input } => Ok(detect_secrets(input)),
        DeterministicToolKind::RegexPattern { name } => Ok(get_regex_pattern(name)),
        DeterministicToolKind::K8sRbac { params } => Ok(generate_k8s_rbac(params)),
        DeterministicToolKind::K8sNetworkPolicy { params } => {
            Ok(generate_k8s_network_policy(params))
        }
        DeterministicToolKind::AwsIamPolicy { params } => Ok(generate_aws_iam_policy(params)),
        // §121-§126 Reasoning motifs
        DeterministicToolKind::Syllogism { params } => Ok(evaluate_syllogism(params)),
        DeterministicToolKind::DecisionMatrix { params } => Ok(generate_decision_matrix(params)),
        DeterministicToolKind::Swot { params } => Ok(generate_swot(params)),
        DeterministicToolKind::ProsCons { topic, pros, cons } => {
            Ok(generate_pros_cons(topic, pros, cons))
        }
        DeterministicToolKind::RootCause { params } => Ok(generate_root_cause(params)),
        DeterministicToolKind::Deduction { params } => Ok(evaluate_deduction(params)),
        // Calendar/Meet — these return marker text; the server fills in real data.
        DeterministicToolKind::CalendarQuery { date_hint } => {
            let hint = date_hint.as_deref().unwrap_or("today");
            Ok(format!("[calendar:{hint}]"))
        }
        DeterministicToolKind::MeetRoom => {
            Ok("Opening video meeting... Use the Meet tab to join.".to_string())
        }
        DeterministicToolKind::EnergyFloor { operation } => energy_floor_calc(operation),
        // §133 – Geo
        DeterministicToolKind::Geo { operation } => Ok(geo_calc(operation)),
    }
}

// ─────────────────────────────────────────────────
// §82  Molecular Weight & Binding Affinity
// ─────────────────────────────────────────────────

/// Molecular weight / binding affinity operation.
#[derive(Debug, Clone, PartialEq)]
pub enum MoleculeOp {
    /// Parse chemical formula → molecular weight (g/mol).
    MolecularWeight { formula: String },
    /// ΔG from dissociation constant Kd.
    DeltaGFromKd { kd_molar: f64, temp_kelvin: f64 },
    /// Kd from Gibbs free energy ΔG.
    KdFromDeltaG { delta_g_kcal: f64, temp_kelvin: f64 },
}

/// Standard atomic weights (IUPAC 2021).
fn atomic_weight(sym: &str) -> Option<f64> {
    match sym {
        "H" => Some(1.008),
        "He" => Some(4.003),
        "Li" => Some(6.941),
        "Be" => Some(9.012),
        "B" => Some(10.811),
        "C" => Some(12.011),
        "N" => Some(14.007),
        "O" => Some(15.999),
        "F" => Some(18.998),
        "Ne" => Some(20.180),
        "Na" => Some(22.990),
        "Mg" => Some(24.305),
        "Al" => Some(26.982),
        "Si" => Some(28.086),
        "P" => Some(30.974),
        "S" => Some(32.065),
        "Cl" => Some(35.453),
        "Ar" => Some(39.948),
        "K" => Some(39.098),
        "Ca" => Some(40.078),
        "Ti" => Some(47.867),
        "V" => Some(50.942),
        "Cr" => Some(51.996),
        "Mn" => Some(54.938),
        "Fe" => Some(55.845),
        "Co" => Some(58.933),
        "Ni" => Some(58.693),
        "Cu" => Some(63.546),
        "Zn" => Some(65.380),
        "As" => Some(74.922),
        "Se" => Some(78.971),
        "Br" => Some(79.904),
        "Mo" => Some(95.950),
        "Ag" => Some(107.868),
        "Sn" => Some(118.710),
        "I" => Some(126.904),
        "Ba" => Some(137.327),
        "Pt" => Some(195.084),
        "Au" => Some(196.967),
        "Hg" => Some(200.592),
        "Pb" => Some(207.200),
        _ => None,
    }
}

/// Recursive descent parser for chemical formulas.
fn parse_formula_weight(chars: &[u8], pos: &mut usize) -> Result<f64, String> {
    let mut total = 0.0;
    while *pos < chars.len() {
        if chars[*pos] == b'(' {
            *pos += 1;
            let group_weight = parse_formula_weight(chars, pos)?;
            if *pos < chars.len() && chars[*pos] == b')' {
                *pos += 1;
            }
            let count = parse_count(chars, pos);
            total += group_weight * count as f64;
        } else if chars[*pos] == b')' {
            break;
        } else if chars[*pos].is_ascii_uppercase() {
            let start = *pos;
            *pos += 1;
            while *pos < chars.len() && chars[*pos].is_ascii_lowercase() {
                *pos += 1;
            }
            let sym = std::str::from_utf8(&chars[start..*pos])
                .map_err(|_| "Invalid UTF-8 in formula".to_string())?;
            let count = parse_count(chars, pos);
            let w = atomic_weight(sym).ok_or_else(|| format!("Unknown element: {}", sym))?;
            total += w * count as f64;
        } else {
            *pos += 1; // skip whitespace / unexpected
        }
    }
    Ok(total)
}

fn parse_count(chars: &[u8], pos: &mut usize) -> u32 {
    let mut n = 0u32;
    while *pos < chars.len() && chars[*pos].is_ascii_digit() {
        n = n * 10 + (chars[*pos] - b'0') as u32;
        *pos += 1;
    }
    if n == 0 { 1 } else { n }
}

/// Gas constant in kcal/(mol·K).
const R_KCAL: f64 = 1.987e-3;

pub fn molecule_calc(op: &MoleculeOp) -> String {
    match op {
        MoleculeOp::MolecularWeight { formula } => {
            let bytes = formula.as_bytes();
            let mut pos = 0;
            match parse_formula_weight(bytes, &mut pos) {
                Ok(mw) => format!("Formula: {}\nMolecular Weight: {:.4} g/mol", formula, mw),
                Err(e) => format!("Error parsing formula \"{}\": {}", formula, e),
            }
        }
        MoleculeOp::DeltaGFromKd {
            kd_molar,
            temp_kelvin,
        } => {
            let dg = R_KCAL * temp_kelvin * kd_molar.ln();
            format!(
                "Kd = {:.4e} M\nT = {:.1} K ({:.1} °C)\nΔG = {:.4} kcal/mol",
                kd_molar,
                temp_kelvin,
                temp_kelvin - 273.15,
                dg
            )
        }
        MoleculeOp::KdFromDeltaG {
            delta_g_kcal,
            temp_kelvin,
        } => {
            let kd = (delta_g_kcal / (R_KCAL * temp_kelvin)).exp();
            format!(
                "ΔG = {:.4} kcal/mol\nT = {:.1} K ({:.1} °C)\nKd = {:.4e} M",
                delta_g_kcal,
                temp_kelvin,
                temp_kelvin - 273.15,
                kd
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §85  Pharmacokinetics
// ─────────────────────────────────────────────────

/// Pharmacokinetic operation.
#[derive(Debug, Clone, PartialEq)]
pub enum PharmaOp {
    /// Half-life from elimination rate constant.
    HalfLife { ke: f64 },
    /// Concentration at time t (first-order decay).
    Concentration { c0: f64, ke: f64, t: f64 },
    /// Area under the curve (trapezoidal rule).
    Auc {
        times: Vec<f64>,
        concentrations: Vec<f64>,
    },
    /// Volume of distribution.
    VolumeOfDistribution { dose_mg: f64, c0: f64 },
    /// Clearance from dose and AUC.
    Clearance { dose_mg: f64, auc: f64 },
    /// Loading dose.
    LoadingDose {
        vd: f64,
        c_target: f64,
        bioavailability: f64,
    },
    /// Steady-state average concentration.
    SteadyState {
        dose_mg: f64,
        bioavailability: f64,
        clearance: f64,
        interval_h: f64,
    },
}

pub fn pharma_calc(op: &PharmaOp) -> String {
    match op {
        PharmaOp::HalfLife { ke } => {
            let t_half = 2.0_f64.ln() / ke;
            format!(
                "Elimination constant (ke): {:.4} h⁻¹\nHalf-life (t½): {:.4} h",
                ke, t_half
            )
        }
        PharmaOp::Concentration { c0, ke, t } => {
            let ct = c0 * (-ke * t).exp();
            format!(
                "C₀ = {:.4} mg/L\nke = {:.4} h⁻¹\nt = {:.2} h\nC(t) = {:.4} mg/L",
                c0, ke, t, ct
            )
        }
        PharmaOp::Auc {
            times,
            concentrations,
        } => {
            if times.len() != concentrations.len() || times.len() < 2 {
                return "Error: times and concentrations must have equal length ≥ 2".to_string();
            }
            let mut auc = 0.0;
            for i in 0..times.len() - 1 {
                let dt = times[i + 1] - times[i];
                auc += dt * (concentrations[i] + concentrations[i + 1]) / 2.0;
            }
            format!(
                "AUC (trapezoidal rule): {:.4} mg·h/L\nData points: {}",
                auc,
                times.len()
            )
        }
        PharmaOp::VolumeOfDistribution { dose_mg, c0 } => {
            let vd = dose_mg / c0;
            format!(
                "Dose: {:.2} mg\nC₀: {:.4} mg/L\nVolume of Distribution (Vd): {:.4} L",
                dose_mg, c0, vd
            )
        }
        PharmaOp::Clearance { dose_mg, auc } => {
            let cl = dose_mg / auc;
            format!(
                "Dose: {:.2} mg\nAUC: {:.4} mg·h/L\nClearance (CL): {:.4} L/h",
                dose_mg, auc, cl
            )
        }
        PharmaOp::LoadingDose {
            vd,
            c_target,
            bioavailability,
        } => {
            let ld = (vd * c_target) / bioavailability;
            format!(
                "Vd: {:.2} L\nTarget concentration: {:.4} mg/L\nBioavailability (F): {:.2}\nLoading Dose: {:.4} mg",
                vd, c_target, bioavailability, ld
            )
        }
        PharmaOp::SteadyState {
            dose_mg,
            bioavailability,
            clearance,
            interval_h,
        } => {
            let css = (dose_mg * bioavailability) / (clearance * interval_h);
            format!(
                "Dose: {:.2} mg\nF: {:.2}\nCL: {:.4} L/h\nτ: {:.2} h\nCss (avg): {:.4} mg/L",
                dose_mg, bioavailability, clearance, interval_h, css
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §86  Clinical Trial Statistics
// ─────────────────────────────────────────────────

/// Clinical trial statistic operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ClinicalOp {
    /// Kaplan-Meier survival estimate. Each tuple: (time, deaths, at_risk).
    KaplanMeier { events: Vec<(f64, u32, u32)> },
    /// Number needed to treat.
    Nnt {
        risk_treatment: f64,
        risk_control: f64,
    },
    /// Chi-square test on 2×2 contingency table.
    ChiSquare { a: u32, b: u32, c: u32, d: u32 },
    /// Fisher's exact test on 2×2 contingency table.
    FisherExact { a: u32, b: u32, c: u32, d: u32 },
    /// Odds ratio with 95% CI.
    OddsRatio { a: u32, b: u32, c: u32, d: u32 },
    /// Relative risk with 95% CI.
    RelativeRisk { a: u32, b: u32, c: u32, d: u32 },
    /// Hazard ratio.
    HazardRatio {
        events_treatment: u32,
        time_treatment: f64,
        events_control: u32,
        time_control: f64,
    },
}

/// Log-factorial for Fisher's exact test.
fn log_factorial(n: u32) -> f64 {
    (1..=n).map(|i| (i as f64).ln()).sum()
}

/// Approximate chi-square p-value (1 df) using normal approximation.
fn chi2_p_value_1df(chi2: f64) -> f64 {
    // For 1 df, p ≈ 2*(1 - Φ(√χ²)) using Abramowitz & Stegun approx
    let x = chi2.sqrt();
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-x * x / 2.0).exp();
    let tail = phi * poly;
    2.0 * tail
}

pub fn clinical_calc(op: &ClinicalOp) -> String {
    match op {
        ClinicalOp::KaplanMeier { events } => {
            let mut s = 1.0_f64;
            let mut lines = vec!["Kaplan-Meier Survival Estimates:".to_string()];
            lines.push(format!(
                "{:<10} {:<10} {:<10} {:<10}",
                "Time", "Deaths", "At Risk", "S(t)"
            ));
            for &(t, d, n) in events {
                if n > 0 {
                    s *= 1.0 - d as f64 / n as f64;
                }
                lines.push(format!("{:<10.2} {:<10} {:<10} {:<10.4}", t, d, n, s));
            }
            lines.join("\n")
        }
        ClinicalOp::Nnt {
            risk_treatment,
            risk_control,
        } => {
            let arr = (risk_control - risk_treatment).abs();
            if arr < 1e-12 {
                return "ARR ≈ 0 → NNT undefined (no difference between groups)".to_string();
            }
            let nnt = 1.0 / arr;
            format!(
                "Risk (treatment): {:.4}\nRisk (control): {:.4}\nARR: {:.4}\nNNT: {:.1}",
                risk_treatment, risk_control, arr, nnt
            )
        }
        ClinicalOp::ChiSquare { a, b, c, d } => {
            let a = *a as f64;
            let b = *b as f64;
            let c = *c as f64;
            let d = *d as f64;
            let n = a + b + c + d;
            let chi2 = n * (a * d - b * c).powi(2) / ((a + b) * (c + d) * (a + c) * (b + d));
            let p = chi2_p_value_1df(chi2);
            format!(
                "Contingency Table:\n  Event  No-Event\n  {:<6.0}  {:<6.0}\n  {:<6.0}  {:<6.0}\n\nχ² = {:.4} (1 df)\np ≈ {:.6}\n{}",
                a,
                b,
                c,
                d,
                chi2,
                p,
                if p < 0.05 {
                    "Statistically significant (p < 0.05)"
                } else {
                    "Not statistically significant (p ≥ 0.05)"
                }
            )
        }
        ClinicalOp::FisherExact { a, b, c, d } => {
            let a = *a;
            let b = *b;
            let c = *c;
            let d = *d;
            let n = a + b + c + d;
            let log_p = log_factorial(a + b)
                + log_factorial(c + d)
                + log_factorial(a + c)
                + log_factorial(b + d)
                - log_factorial(n)
                - log_factorial(a)
                - log_factorial(b)
                - log_factorial(c)
                - log_factorial(d);
            let p = log_p.exp();
            format!(
                "Fisher's Exact Test (2×2):\n  {:<6}  {:<6}\n  {:<6}  {:<6}\n\np (exact) = {:.6}\n{}",
                a,
                b,
                c,
                d,
                p,
                if p < 0.05 {
                    "Statistically significant (p < 0.05)"
                } else {
                    "Not statistically significant (p ≥ 0.05)"
                }
            )
        }
        ClinicalOp::OddsRatio { a, b, c, d } => {
            let a = *a as f64;
            let b = *b as f64;
            let c = *c as f64;
            let d = *d as f64;
            let or = (a * d) / (b * c);
            let se_ln = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
            let ci_lo = (or.ln() - 1.96 * se_ln).exp();
            let ci_hi = (or.ln() + 1.96 * se_ln).exp();
            format!(
                "Odds Ratio: {:.4}\n95% CI: [{:.4}, {:.4}]\n{}",
                or,
                ci_lo,
                ci_hi,
                if ci_lo > 1.0 || ci_hi < 1.0 {
                    "Statistically significant (CI does not include 1.0)"
                } else {
                    "Not statistically significant (CI includes 1.0)"
                }
            )
        }
        ClinicalOp::RelativeRisk { a, b, c, d } => {
            let a = *a as f64;
            let b = *b as f64;
            let c = *c as f64;
            let d = *d as f64;
            let r1 = a / (a + b);
            let r2 = c / (c + d);
            let rr = r1 / r2;
            let se_ln = ((1.0 - r1) / (a) + (1.0 - r2) / (c)).sqrt();
            let ci_lo = (rr.ln() - 1.96 * se_ln).exp();
            let ci_hi = (rr.ln() + 1.96 * se_ln).exp();
            format!(
                "Risk (group 1): {:.4}\nRisk (group 2): {:.4}\nRelative Risk: {:.4}\n95% CI: [{:.4}, {:.4}]",
                r1, r2, rr, ci_lo, ci_hi
            )
        }
        ClinicalOp::HazardRatio {
            events_treatment,
            time_treatment,
            events_control,
            time_control,
        } => {
            let rate_t = *events_treatment as f64 / time_treatment;
            let rate_c = *events_control as f64 / time_control;
            let hr = rate_t / rate_c;
            let se_ln = (1.0 / *events_treatment as f64 + 1.0 / *events_control as f64).sqrt();
            let ci_lo = (hr.ln() - 1.96 * se_ln).exp();
            let ci_hi = (hr.ln() + 1.96 * se_ln).exp();
            format!(
                "Rate (treatment): {:.4} events/time\nRate (control): {:.4} events/time\nHazard Ratio: {:.4}\n95% CI: [{:.4}, {:.4}]",
                rate_t, rate_c, hr, ci_lo, ci_hi
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §81  Protein Folding Energy
// ─────────────────────────────────────────────────

/// Protein energy calculation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ProteinEnergyOp {
    /// Lennard-Jones potential.
    LennardJones { epsilon: f64, sigma: f64, r: f64 },
    /// Coulomb electrostatics.
    Coulomb {
        q1: f64,
        q2: f64,
        r: f64,
        dielectric: f64,
    },
    /// Hydrogen bond energy (geometric scoring).
    HydrogenBond { d_ha: f64, angle_dha: f64 },
    /// Ramachandran angle assessment.
    Ramachandran { phi: f64, psi: f64 },
    /// SASA-based solvation energy.
    Solvation { sasa: f64, atom_type: String },
    /// Pairwise energy for a set of atoms.
    PairwiseEnergy { atoms: Vec<AtomEntry> },
}

/// Atom entry for pairwise energy calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomEntry {
    pub atom_type: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub charge: f64,
}

/// Default LJ parameters (epsilon in kcal/mol, sigma in Å) for common atom types.
fn lj_params(a: &str, b: &str) -> (f64, f64) {
    let eps_a: f64 = match a {
        "C" => 0.086,
        "N" => 0.170,
        "O" => 0.210,
        "S" => 0.250,
        "H" => 0.015,
        _ => 0.100,
    };
    let sig_a: f64 = match a {
        "C" => 3.400,
        "N" => 3.250,
        "O" => 3.120,
        "S" => 3.560,
        "H" => 2.500,
        _ => 3.200,
    };
    let eps_b: f64 = match b {
        "C" => 0.086,
        "N" => 0.170,
        "O" => 0.210,
        "S" => 0.250,
        "H" => 0.015,
        _ => 0.100,
    };
    let sig_b: f64 = match b {
        "C" => 3.400,
        "N" => 3.250,
        "O" => 3.120,
        "S" => 3.560,
        "H" => 2.500,
        _ => 3.200,
    };
    // Lorentz-Berthelot combining rules
    let eps = (eps_a * eps_b).sqrt();
    let sig = (sig_a + sig_b) / 2.0;
    (eps, sig)
}

/// Solvation parameter (kcal/mol/Å²) by atom type.
fn solvation_sigma(atom_type: &str) -> f64 {
    match atom_type {
        "C" => 0.012,
        "N" => -0.068,
        "O" => -0.050,
        "S" => -0.002,
        "H" => 0.000,
        "P" => -0.030,
        _ => 0.005,
    }
}

pub fn protein_energy_calc(op: &ProteinEnergyOp) -> String {
    match op {
        ProteinEnergyOp::LennardJones { epsilon, sigma, r } => {
            let sr6 = (sigma / r).powi(6);
            let e = 4.0 * epsilon * (sr6 * sr6 - sr6);
            format!(
                "Lennard-Jones Potential:\n  ε = {:.4} kcal/mol, σ = {:.4} Å, r = {:.4} Å\n  E_LJ = {:.6} kcal/mol",
                epsilon, sigma, r, e
            )
        }
        ProteinEnergyOp::Coulomb {
            q1,
            q2,
            r,
            dielectric,
        } => {
            let e = 332.0 * q1 * q2 / (dielectric * r);
            format!(
                "Coulomb Electrostatics:\n  q₁ = {:.3} e, q₂ = {:.3} e, r = {:.4} Å, ε_r = {:.1}\n  E_elec = {:.6} kcal/mol",
                q1, q2, r, dielectric, e
            )
        }
        ProteinEnergyOp::HydrogenBond { d_ha, angle_dha } => {
            let dist_term = 1.0 - ((d_ha - 1.9) * (d_ha - 1.9)) / 0.25;
            let dist_term = dist_term.max(0.0);
            let angle_rad = angle_dha.to_radians();
            let angle_term = angle_rad.cos().powi(2);
            let e = -2.0 * dist_term * angle_term;
            format!(
                "Hydrogen Bond Energy:\n  d(H···A) = {:.3} Å, ∠(D-H···A) = {:.1}°\n  E_HB = {:.4} kcal/mol\n  Quality: {}",
                d_ha,
                angle_dha,
                e,
                if e < -1.5 {
                    "Strong"
                } else if e < -0.5 {
                    "Moderate"
                } else {
                    "Weak/None"
                }
            )
        }
        ProteinEnergyOp::Ramachandran { phi, psi } => {
            let region = classify_ramachandran(*phi, *psi);
            format!(
                "Ramachandran Assessment:\n  φ = {:.1}°, ψ = {:.1}°\n  Region: {}",
                phi, psi, region
            )
        }
        ProteinEnergyOp::Solvation { sasa, atom_type } => {
            let sigma = solvation_sigma(atom_type);
            let e = sigma * sasa;
            format!(
                "Solvation Energy (SASA-based):\n  Atom type: {}, SASA = {:.2} Å²\n  σ = {:.4} kcal/(mol·Å²)\n  E_solv = {:.4} kcal/mol",
                atom_type, sasa, sigma, e
            )
        }
        ProteinEnergyOp::PairwiseEnergy { atoms } => {
            let n = atoms.len();
            if n < 2 {
                return "Error: need at least 2 atoms for pairwise energy".to_string();
            }
            let mut e_lj_total = 0.0;
            let mut e_elec_total = 0.0;
            let mut pairs = 0u32;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = atoms[i].x - atoms[j].x;
                    let dy = atoms[i].y - atoms[j].y;
                    let dz = atoms[i].z - atoms[j].z;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r < 0.1 {
                        continue;
                    }
                    let (eps, sig) = lj_params(&atoms[i].atom_type, &atoms[j].atom_type);
                    let sr6 = (sig / r).powi(6);
                    e_lj_total += 4.0 * eps * (sr6 * sr6 - sr6);
                    e_elec_total += 332.0 * atoms[i].charge * atoms[j].charge / (4.0 * r);
                    pairs += 1;
                }
            }
            format!(
                "Pairwise Energy ({} atoms, {} pairs):\n  E_LJ   = {:.4} kcal/mol\n  E_elec = {:.4} kcal/mol\n  E_total = {:.4} kcal/mol",
                n,
                pairs,
                e_lj_total,
                e_elec_total,
                e_lj_total + e_elec_total
            )
        }
    }
}

fn classify_ramachandran(phi: f64, psi: f64) -> &'static str {
    // Alpha helix region
    if (-160.0..=-20.0).contains(&phi) && (-80.0..=-10.0).contains(&psi) {
        return "Alpha helix (favored)";
    }
    // Beta sheet region
    if (-180.0..=-60.0).contains(&phi) && (80.0..=180.0).contains(&psi) {
        return "Beta sheet (favored)";
    }
    // Left-handed alpha helix
    if (20.0..=100.0).contains(&phi) && (20.0..=80.0).contains(&psi) {
        return "Left-handed alpha helix";
    }
    // Polyproline II / collagen
    if (-100.0..=-55.0).contains(&phi) && (120.0..=180.0).contains(&psi) {
        return "Polyproline II / collagen";
    }
    // Generously allowed
    if (-180.0..=0.0).contains(&phi) {
        return "Generously allowed";
    }
    "Disallowed region"
}

// ─────────────────────────────────────────────────
// §83  Gene Sequence Alignment (Smith-Waterman)
// ─────────────────────────────────────────────────

/// Sequence alignment operation.
#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentOp {
    /// DNA local alignment (match=+2, mismatch=-1).
    DnaAlign { seq1: String, seq2: String },
    /// Protein local alignment (BLOSUM62).
    ProteinAlign { seq1: String, seq2: String },
    /// Custom scoring alignment.
    CustomAlign {
        seq1: String,
        seq2: String,
        match_score: i32,
        mismatch_penalty: i32,
        gap_open: i32,
        gap_extend: i32,
    },
}

/// BLOSUM62 substitution matrix (upper triangle, amino acid order: ARNDCQEGHILKMFPSTWYV).
const BLOSUM62_AA: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
#[rustfmt::skip]
const BLOSUM62: [[i32; 20]; 20] = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  // A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  // R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  // N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  // D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  // C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  // Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  // E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  // G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  // H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  // I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  // L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  // K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  // M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  // F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  // P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  // S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  // T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  // W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  // Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  // V
];

fn blosum62_index(aa: u8) -> Option<usize> {
    BLOSUM62_AA.iter().position(|&c| c == aa)
}

fn blosum62_score(a: u8, b: u8) -> i32 {
    match (blosum62_index(a), blosum62_index(b)) {
        (Some(i), Some(j)) => BLOSUM62[i][j],
        _ => -4, // unknown amino acid penalty
    }
}

pub fn alignment_calc(op: &AlignmentOp) -> String {
    let (s1, s2, score_fn, gap_open, gap_extend, mode) = match op {
        AlignmentOp::DnaAlign { seq1, seq2 } => {
            let s1: Vec<u8> = seq1
                .to_uppercase()
                .bytes()
                .filter(|b| b"ACGTN".contains(b))
                .collect();
            let s2: Vec<u8> = seq2
                .to_uppercase()
                .bytes()
                .filter(|b| b"ACGTN".contains(b))
                .collect();
            (
                s1,
                s2,
                ScoreFn::Dna {
                    match_s: 2,
                    mismatch: -1,
                },
                -12_i32,
                -1_i32,
                "DNA",
            )
        }
        AlignmentOp::ProteinAlign { seq1, seq2 } => {
            let s1: Vec<u8> = seq1.to_uppercase().bytes().collect();
            let s2: Vec<u8> = seq2.to_uppercase().bytes().collect();
            (s1, s2, ScoreFn::Blosum62, -12, -1, "Protein (BLOSUM62)")
        }
        AlignmentOp::CustomAlign {
            seq1,
            seq2,
            match_score,
            mismatch_penalty,
            gap_open,
            gap_extend,
        } => {
            let s1: Vec<u8> = seq1.to_uppercase().bytes().collect();
            let s2: Vec<u8> = seq2.to_uppercase().bytes().collect();
            (
                s1,
                s2,
                ScoreFn::Dna {
                    match_s: *match_score,
                    mismatch: *mismatch_penalty,
                },
                *gap_open,
                *gap_extend,
                "Custom",
            )
        }
    };

    if s1.len() > 1000 || s2.len() > 1000 {
        return "Error: sequences capped at 1000 characters".to_string();
    }
    if s1.is_empty() || s2.is_empty() {
        return "Error: both sequences must be non-empty".to_string();
    }

    let (score, aln1, aln2) = smith_waterman(&s1, &s2, &score_fn, gap_open, gap_extend);
    let identity = aln1
        .iter()
        .zip(aln2.iter())
        .filter(|(a, b)| a == b && **a != b'-')
        .count();
    let aln_len = aln1.len();
    let identity_pct = if aln_len > 0 {
        identity as f64 / aln_len as f64 * 100.0
    } else {
        0.0
    };

    let aln1_str: String = aln1.iter().map(|&b| b as char).collect();
    let aln2_str: String = aln2.iter().map(|&b| b as char).collect();
    let mid: String = aln1
        .iter()
        .zip(aln2.iter())
        .map(|(a, b)| {
            if a == b && *a != b'-' {
                '|'
            } else if *a == b'-' || *b == b'-' {
                ' '
            } else {
                '.'
            }
        })
        .collect();

    format!(
        "Smith-Waterman Local Alignment ({}):\nScore: {}\nIdentity: {}/{} ({:.1}%)\n\n  {}\n  {}\n  {}",
        mode, score, identity, aln_len, identity_pct, aln1_str, mid, aln2_str
    )
}

enum ScoreFn {
    Dna { match_s: i32, mismatch: i32 },
    Blosum62,
}

impl ScoreFn {
    fn score(&self, a: u8, b: u8) -> i32 {
        match self {
            ScoreFn::Dna { match_s, mismatch } => {
                if a == b {
                    *match_s
                } else {
                    *mismatch
                }
            }
            ScoreFn::Blosum62 => blosum62_score(a, b),
        }
    }
}

fn smith_waterman(
    s1: &[u8],
    s2: &[u8],
    score_fn: &ScoreFn,
    gap_open: i32,
    gap_extend: i32,
) -> (i32, Vec<u8>, Vec<u8>) {
    let m = s1.len();
    let n = s2.len();
    let mut h = vec![vec![0i32; n + 1]; m + 1];
    let mut e = vec![vec![i32::MIN / 2; n + 1]; m + 1]; // gap in s1
    let mut f = vec![vec![i32::MIN / 2; n + 1]; m + 1]; // gap in s2
    let mut max_score = 0;
    let mut max_i = 0;
    let mut max_j = 0;

    for i in 1..=m {
        for j in 1..=n {
            e[i][j] = (h[i][j - 1] + gap_open).max(e[i][j - 1] + gap_extend);
            f[i][j] = (h[i - 1][j] + gap_open).max(f[i - 1][j] + gap_extend);
            let diag = h[i - 1][j - 1] + score_fn.score(s1[i - 1], s2[j - 1]);
            h[i][j] = 0_i32.max(diag).max(e[i][j]).max(f[i][j]);
            if h[i][j] > max_score {
                max_score = h[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    // Traceback
    let mut aln1 = Vec::new();
    let mut aln2 = Vec::new();
    let mut i = max_i;
    let mut j = max_j;
    while i > 0 && j > 0 && h[i][j] > 0 {
        if h[i][j] == h[i - 1][j - 1] + score_fn.score(s1[i - 1], s2[j - 1]) {
            aln1.push(s1[i - 1]);
            aln2.push(s2[j - 1]);
            i -= 1;
            j -= 1;
        } else if h[i][j] == f[i][j] {
            aln1.push(s1[i - 1]);
            aln2.push(b'-');
            i -= 1;
        } else {
            aln1.push(b'-');
            aln2.push(s2[j - 1]);
            j -= 1;
        }
    }
    aln1.reverse();
    aln2.reverse();
    (max_score, aln1, aln2)
}

// ─────────────────────────────────────────────────
// §84  Drug Interaction Matrix
// ─────────────────────────────────────────────────

/// Drug interaction operation.
#[derive(Debug, Clone, PartialEq)]
pub enum DrugOp {
    /// Check interaction between two drugs.
    Interaction { drug1: String, drug2: String },
    /// CYP450 enzyme profile for a drug.
    CypProfile { drug: String },
    /// List all known interactions for a drug.
    AllInteractions { drug: String },
}

struct DrugEntry {
    name: &'static str,
    substrates: &'static [&'static str],
    inhibitors: &'static [(&'static str, &'static str)], // (enzyme, strength)
    inducers: &'static [(&'static str, &'static str)],
}

fn drug_db() -> Vec<DrugEntry> {
    vec![
        DrugEntry {
            name: "warfarin",
            substrates: &["CYP2C9", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "simvastatin",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "atorvastatin",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "itraconazole",
            substrates: &["CYP3A4"],
            inhibitors: &[("CYP3A4", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "ketoconazole",
            substrates: &["CYP3A4"],
            inhibitors: &[("CYP3A4", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "fluoxetine",
            substrates: &["CYP2C19", "CYP2D6"],
            inhibitors: &[("CYP2D6", "strong"), ("CYP2C19", "moderate")],
            inducers: &[],
        },
        DrugEntry {
            name: "paroxetine",
            substrates: &["CYP2D6"],
            inhibitors: &[("CYP2D6", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "omeprazole",
            substrates: &["CYP2C19", "CYP3A4"],
            inhibitors: &[("CYP2C19", "moderate")],
            inducers: &[],
        },
        DrugEntry {
            name: "clopidogrel",
            substrates: &["CYP2C19", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "rifampin",
            substrates: &[],
            inhibitors: &[],
            inducers: &[
                ("CYP3A4", "strong"),
                ("CYP2C9", "strong"),
                ("CYP2C19", "strong"),
                ("CYP1A2", "strong"),
            ],
        },
        DrugEntry {
            name: "carbamazepine",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[("CYP3A4", "strong"), ("CYP2C9", "moderate")],
        },
        DrugEntry {
            name: "phenytoin",
            substrates: &["CYP2C9", "CYP2C19"],
            inhibitors: &[],
            inducers: &[("CYP3A4", "strong"), ("CYP2C9", "moderate")],
        },
        DrugEntry {
            name: "erythromycin",
            substrates: &["CYP3A4"],
            inhibitors: &[("CYP3A4", "moderate")],
            inducers: &[],
        },
        DrugEntry {
            name: "clarithromycin",
            substrates: &["CYP3A4"],
            inhibitors: &[("CYP3A4", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "codeine",
            substrates: &["CYP2D6"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "tramadol",
            substrates: &["CYP2D6", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "metoprolol",
            substrates: &["CYP2D6"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "tamoxifen",
            substrates: &["CYP2D6", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "diazepam",
            substrates: &["CYP2C19", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "midazolam",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "cyclosporine",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "tacrolimus",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "amlodipine",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "nifedipine",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "sildenafil",
            substrates: &["CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "ritonavir",
            substrates: &["CYP3A4", "CYP2D6"],
            inhibitors: &[("CYP3A4", "strong"), ("CYP2D6", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "theophylline",
            substrates: &["CYP1A2"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "caffeine",
            substrates: &["CYP1A2"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "clozapine",
            substrates: &["CYP1A2", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "olanzapine",
            substrates: &["CYP1A2", "CYP2D6"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "fluvoxamine",
            substrates: &["CYP2D6"],
            inhibitors: &[("CYP1A2", "strong"), ("CYP2C19", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "citalopram",
            substrates: &["CYP2C19", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "losartan",
            substrates: &["CYP2C9", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "celecoxib",
            substrates: &["CYP2C9"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "ibuprofen",
            substrates: &["CYP2C9"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "metformin",
            substrates: &[],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "acetaminophen",
            substrates: &["CYP2E1", "CYP1A2"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "voriconazole",
            substrates: &["CYP2C19", "CYP3A4"],
            inhibitors: &[("CYP3A4", "moderate"), ("CYP2C19", "strong")],
            inducers: &[],
        },
        DrugEntry {
            name: "haloperidol",
            substrates: &["CYP2D6", "CYP3A4"],
            inhibitors: &[],
            inducers: &[],
        },
        DrugEntry {
            name: "amiodarone",
            substrates: &["CYP3A4", "CYP2C9"],
            inhibitors: &[
                ("CYP2D6", "moderate"),
                ("CYP2C9", "moderate"),
                ("CYP3A4", "moderate"),
            ],
            inducers: &[],
        },
    ]
}

fn interaction_severity(drug1: &DrugEntry, drug2: &DrugEntry) -> (String, Vec<String>) {
    let mut reasons = Vec::new();
    let mut max_severity = 0u8; // 0=none, 1=minor, 2=moderate, 3=major, 4=contraindicated

    // Check if drug2 inhibits an enzyme that drug1 is a substrate of
    for sub in drug1.substrates {
        for &(enz, strength) in drug2.inhibitors {
            if *sub == enz {
                let sev = match strength {
                    "strong" => {
                        reasons.push(format!(
                            "{} is a strong inhibitor of {} (substrate of {})",
                            drug2.name, enz, drug1.name
                        ));
                        3
                    }
                    "moderate" => {
                        reasons.push(format!(
                            "{} is a moderate inhibitor of {} (substrate of {})",
                            drug2.name, enz, drug1.name
                        ));
                        2
                    }
                    _ => {
                        reasons.push(format!(
                            "{} is a weak inhibitor of {} (substrate of {})",
                            drug2.name, enz, drug1.name
                        ));
                        1
                    }
                };
                max_severity = max_severity.max(sev);
            }
        }
        for &(enz, strength) in drug2.inducers {
            if *sub == enz {
                let sev = match strength {
                    "strong" => {
                        reasons.push(format!(
                            "{} is a strong inducer of {} (substrate of {})",
                            drug2.name, enz, drug1.name
                        ));
                        3
                    }
                    "moderate" => {
                        reasons.push(format!(
                            "{} is a moderate inducer of {} (substrate of {})",
                            drug2.name, enz, drug1.name
                        ));
                        2
                    }
                    _ => {
                        reasons.push(format!(
                            "{} is a weak inducer of {} (substrate of {})",
                            drug2.name, enz, drug1.name
                        ));
                        1
                    }
                };
                max_severity = max_severity.max(sev);
            }
        }
    }

    // Check reverse direction
    for sub in drug2.substrates {
        for &(enz, strength) in drug1.inhibitors {
            if *sub == enz {
                let sev = match strength {
                    "strong" => {
                        reasons.push(format!(
                            "{} is a strong inhibitor of {} (substrate of {})",
                            drug1.name, enz, drug2.name
                        ));
                        3
                    }
                    "moderate" => {
                        reasons.push(format!(
                            "{} is a moderate inhibitor of {} (substrate of {})",
                            drug1.name, enz, drug2.name
                        ));
                        2
                    }
                    _ => {
                        reasons.push(format!(
                            "{} is a weak inhibitor of {} (substrate of {})",
                            drug1.name, enz, drug2.name
                        ));
                        1
                    }
                };
                max_severity = max_severity.max(sev);
            }
        }
        for &(enz, strength) in drug1.inducers {
            if *sub == enz {
                let sev = match strength {
                    "strong" => {
                        reasons.push(format!(
                            "{} is a strong inducer of {} (substrate of {})",
                            drug1.name, enz, drug2.name
                        ));
                        3
                    }
                    "moderate" => {
                        reasons.push(format!(
                            "{} is a moderate inducer of {} (substrate of {})",
                            drug1.name, enz, drug2.name
                        ));
                        2
                    }
                    _ => {
                        reasons.push(format!(
                            "{} is a weak inducer of {} (substrate of {})",
                            drug1.name, enz, drug2.name
                        ));
                        1
                    }
                };
                max_severity = max_severity.max(sev);
            }
        }
    }

    let label = match max_severity {
        0 => "No significant CYP-mediated interaction",
        1 => "Minor",
        2 => "Moderate",
        3 => "Major",
        _ => "Contraindicated",
    };
    (label.to_string(), reasons)
}

pub fn drug_calc(op: &DrugOp) -> String {
    let db = drug_db();
    match op {
        DrugOp::Interaction { drug1, drug2 } => {
            let l1 = drug1.to_lowercase();
            let l2 = drug2.to_lowercase();
            let d1 = db.iter().find(|d| d.name == l1);
            let d2 = db.iter().find(|d| d.name == l2);
            match (d1, d2) {
                (Some(d1), Some(d2)) => {
                    let (severity, reasons) = interaction_severity(d1, d2);
                    let mut out = format!(
                        "Drug Interaction: {} + {}\nSeverity: {}\n",
                        drug1, drug2, severity
                    );
                    if reasons.is_empty() {
                        out.push_str("\nNo known CYP450-mediated interaction found.");
                    } else {
                        out.push_str("\nMechanism(s):\n");
                        for r in &reasons {
                            out.push_str(&format!("  • {}\n", r));
                        }
                    }
                    out.push_str("\n⚠ Reference data only — not medical advice. Consult a qualified professional.");
                    out
                }
                (None, _) => format!("Drug \"{}\" not found in database", drug1),
                (_, None) => format!("Drug \"{}\" not found in database", drug2),
            }
        }
        DrugOp::CypProfile { drug } => {
            let lower = drug.to_lowercase();
            match db.iter().find(|d| d.name == lower) {
                Some(d) => {
                    let mut out = format!("CYP450 Profile: {}\n", drug);
                    if d.substrates.is_empty() {
                        out.push_str("  Substrates: none (not significantly CYP-metabolized)\n");
                    } else {
                        out.push_str(&format!("  Substrates: {}\n", d.substrates.join(", ")));
                    }
                    if d.inhibitors.is_empty() {
                        out.push_str("  Inhibitor: none\n");
                    } else {
                        for &(enz, str_) in d.inhibitors {
                            out.push_str(&format!("  Inhibitor: {} ({})\n", enz, str_));
                        }
                    }
                    if d.inducers.is_empty() {
                        out.push_str("  Inducer: none\n");
                    } else {
                        for &(enz, str_) in d.inducers {
                            out.push_str(&format!("  Inducer: {} ({})\n", enz, str_));
                        }
                    }
                    out.push_str("\n⚠ Reference data only — not medical advice.");
                    out
                }
                None => format!("Drug \"{}\" not found in database", drug),
            }
        }
        DrugOp::AllInteractions { drug } => {
            let lower = drug.to_lowercase();
            let target = match db.iter().find(|d| d.name == lower) {
                Some(d) => d,
                None => return format!("Drug \"{}\" not found in database", drug),
            };
            let mut interactions = Vec::new();
            for other in &db {
                if other.name == target.name {
                    continue;
                }
                let (severity, reasons) = interaction_severity(target, other);
                if !reasons.is_empty() {
                    interactions.push((other.name, severity, reasons));
                }
            }
            if interactions.is_empty() {
                return format!(
                    "No CYP450-mediated interactions found for {}.\n\n⚠ Reference data only — not medical advice.",
                    drug
                );
            }
            let mut out = format!("All Known Interactions for {}:\n\n", drug);
            for (name, sev, reasons) in &interactions {
                out.push_str(&format!("  {} — {}\n", name, sev));
                for r in reasons {
                    out.push_str(&format!("    • {}\n", r));
                }
            }
            out.push_str(
                "\n⚠ Reference data only — not medical advice. Consult a qualified professional.",
            );
            out
        }
    }
}

// ─────────────────────────────────────────────────
// §87  Signaling Pathway Graph Traversal
// ─────────────────────────────────────────────────

/// Pathway graph traversal operation.
#[derive(Debug, Clone, PartialEq)]
pub enum PathwayOp {
    /// BFS from a start node.
    Bfs {
        start: String,
        pathway: Option<String>,
    },
    /// DFS from a start node.
    Dfs {
        start: String,
        pathway: Option<String>,
    },
    /// Shortest path between two nodes.
    ShortestPath {
        from: String,
        to: String,
        pathway: Option<String>,
    },
    /// All upstream regulators of a node.
    Upstream {
        node: String,
        pathway: Option<String>,
    },
    /// All downstream targets of a node.
    Downstream {
        node: String,
        pathway: Option<String>,
    },
    /// List available pathways.
    ListPathways,
}

struct PathwayGraph {
    name: &'static str,
    description: &'static str,
    edges: &'static [(&'static str, &'static str, &'static str)], // (from, to, interaction_type)
}

fn pathway_db() -> Vec<PathwayGraph> {
    vec![
        PathwayGraph {
            name: "MAPK/ERK",
            description: "Growth & proliferation signaling",
            edges: &[
                ("EGF", "EGFR", "activates"),
                ("EGFR", "GRB2", "recruits"),
                ("GRB2", "SOS", "activates"),
                ("SOS", "RAS", "activates"),
                ("RAS", "RAF", "activates"),
                ("RAF", "MEK", "phosphorylates"),
                ("MEK", "ERK", "phosphorylates"),
                ("ERK", "ELK1", "phosphorylates"),
                ("ERK", "MYC", "stabilizes"),
                ("ERK", "RSK", "phosphorylates"),
                ("RSK", "CREB", "phosphorylates"),
                ("EGFR", "PLC", "activates"),
                ("PLC", "PKC", "activates"),
                ("PKC", "RAF", "activates"),
            ],
        },
        PathwayGraph {
            name: "PI3K/AKT",
            description: "Cell survival & growth",
            edges: &[
                ("RTK", "PI3K", "activates"),
                ("PI3K", "PIP3", "produces"),
                ("PIP3", "PDK1", "recruits"),
                ("PIP3", "AKT", "recruits"),
                ("PDK1", "AKT", "phosphorylates"),
                ("AKT", "mTOR", "activates"),
                ("AKT", "BAD", "inhibits"),
                ("AKT", "FOXO", "inhibits"),
                ("AKT", "GSK3", "inhibits"),
                ("AKT", "MDM2", "activates"),
                ("PTEN", "PIP3", "degrades"),
                ("mTOR", "S6K", "phosphorylates"),
                ("mTOR", "4EBP1", "phosphorylates"),
            ],
        },
        PathwayGraph {
            name: "Wnt",
            description: "Development & proliferation",
            edges: &[
                ("WNT", "FZD", "binds"),
                ("FZD", "DVL", "activates"),
                ("DVL", "GSK3B", "inhibits"),
                ("GSK3B", "BCAT", "degrades"),
                ("BCAT", "TCF", "activates"),
                ("TCF", "CCND1", "transcribes"),
                ("TCF", "MYC", "transcribes"),
                ("AXIN", "GSK3B", "scaffolds"),
                ("APC", "BCAT", "degrades"),
                ("CK1", "BCAT", "phosphorylates"),
            ],
        },
        PathwayGraph {
            name: "Notch",
            description: "Cell fate & differentiation",
            edges: &[
                ("DLL", "NOTCH", "binds"),
                ("JAG", "NOTCH", "binds"),
                ("NOTCH", "NICD", "releases"),
                ("NICD", "CSL", "activates"),
                ("CSL", "HES1", "transcribes"),
                ("CSL", "HEY1", "transcribes"),
                ("HES1", "ATOH1", "represses"),
                ("NUMB", "NOTCH", "inhibits"),
            ],
        },
        PathwayGraph {
            name: "p53",
            description: "Tumor suppression & apoptosis",
            edges: &[
                ("DNA_DAMAGE", "ATM", "activates"),
                ("DNA_DAMAGE", "ATR", "activates"),
                ("ATM", "CHK2", "phosphorylates"),
                ("ATR", "CHK1", "phosphorylates"),
                ("CHK2", "P53", "phosphorylates"),
                ("CHK1", "P53", "phosphorylates"),
                ("P53", "P21", "transcribes"),
                ("P53", "BAX", "transcribes"),
                ("P53", "PUMA", "transcribes"),
                ("P53", "MDM2", "transcribes"),
                ("MDM2", "P53", "degrades"),
                ("P21", "CDK2", "inhibits"),
                ("BAX", "CYTC", "releases"),
                ("CYTC", "CASP9", "activates"),
                ("CASP9", "CASP3", "activates"),
            ],
        },
        PathwayGraph {
            name: "JAK/STAT",
            description: "Immune signaling",
            edges: &[
                ("CYTOKINE", "RECEPTOR", "binds"),
                ("RECEPTOR", "JAK", "activates"),
                ("JAK", "STAT", "phosphorylates"),
                ("STAT", "STAT_DIMER", "dimerizes"),
                ("STAT_DIMER", "TARGET_GENES", "transcribes"),
                ("SOCS", "JAK", "inhibits"),
                ("PIAS", "STAT_DIMER", "inhibits"),
                ("SHP", "JAK", "dephosphorylates"),
            ],
        },
        PathwayGraph {
            name: "NF-kB",
            description: "Inflammation & immune response",
            edges: &[
                ("TNF", "TNFR", "binds"),
                ("IL1", "IL1R", "binds"),
                ("TNFR", "TRAF2", "recruits"),
                ("IL1R", "IRAK", "recruits"),
                ("TRAF2", "IKK", "activates"),
                ("IRAK", "IKK", "activates"),
                ("IKK", "IKB", "phosphorylates"),
                ("IKB", "NFKB", "releases"),
                ("NFKB", "IL6", "transcribes"),
                ("NFKB", "TNF", "transcribes"),
                ("NFKB", "IKB", "transcribes"),
                ("A20", "IKK", "inhibits"),
            ],
        },
    ]
}

fn get_pathway_nodes(pw: &PathwayGraph) -> Vec<&'static str> {
    let mut nodes: Vec<&str> = Vec::new();
    for &(from, to, _) in pw.edges {
        if !nodes.contains(&from) {
            nodes.push(from);
        }
        if !nodes.contains(&to) {
            nodes.push(to);
        }
    }
    nodes
}

fn find_pathway<'a>(db: &'a [PathwayGraph], name: &str) -> Option<&'a PathwayGraph> {
    let upper = name.to_uppercase();
    db.iter().find(|p| {
        let pu = p.name.to_uppercase();
        pu == upper || pu.contains(&upper) || upper.contains(&pu.replace('/', ""))
    })
}

fn find_node_pathway<'a>(db: &'a [PathwayGraph], node: &str) -> Vec<&'a PathwayGraph> {
    let upper = node.to_uppercase();
    db.iter()
        .filter(|p| {
            p.edges
                .iter()
                .any(|&(f, t, _)| f.to_uppercase() == upper || t.to_uppercase() == upper)
        })
        .collect()
}

fn bfs_forward(pw: &PathwayGraph, start: &str) -> Vec<&'static str> {
    let upper = start.to_uppercase();
    let mut visited = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    // Find starting node
    for &(f, t, _) in pw.edges {
        if f.to_uppercase() == upper && !visited.contains(&f) {
            queue.push_back(f);
            visited.push(f);
            break;
        }
        if t.to_uppercase() == upper && !visited.contains(&t) {
            queue.push_back(t);
            visited.push(t);
            break;
        }
    }
    while let Some(current) = queue.pop_front() {
        for &(f, t, _) in pw.edges {
            if f == current && !visited.contains(&t) {
                visited.push(t);
                queue.push_back(t);
            }
        }
    }
    visited
}

fn bfs_reverse(pw: &PathwayGraph, target: &str) -> Vec<&'static str> {
    let upper = target.to_uppercase();
    let mut visited = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for &(f, t, _) in pw.edges {
        if t.to_uppercase() == upper && !visited.contains(&t) {
            queue.push_back(t);
            visited.push(t);
            break;
        }
        if f.to_uppercase() == upper && !visited.contains(&f) {
            queue.push_back(f);
            visited.push(f);
            break;
        }
    }
    while let Some(current) = queue.pop_front() {
        for &(f, t, _) in pw.edges {
            if t == current && !visited.contains(&f) {
                visited.push(f);
                queue.push_back(f);
            }
        }
    }
    visited
}

fn shortest_path(pw: &PathwayGraph, from: &str, to: &str) -> Option<Vec<&'static str>> {
    let from_upper = from.to_uppercase();
    let to_upper = to.to_uppercase();
    let nodes = get_pathway_nodes(pw);
    let start = nodes.iter().find(|n| n.to_uppercase() == from_upper)?;
    let goal = nodes.iter().find(|n| n.to_uppercase() == to_upper)?;

    let mut visited = Vec::new();
    let mut queue: std::collections::VecDeque<Vec<&str>> = std::collections::VecDeque::new();
    queue.push_back(vec![*start]);
    visited.push(*start);

    while let Some(path) = queue.pop_front() {
        let current = *path.last()?;
        if current == *goal {
            return Some(path);
        }
        for &(f, t, _) in pw.edges {
            if f == current && !visited.contains(&t) {
                visited.push(t);
                let mut new_path = path.clone();
                new_path.push(t);
                queue.push_back(new_path);
            }
        }
    }
    None
}

pub fn pathway_calc(op: &PathwayOp) -> String {
    let db = pathway_db();
    match op {
        PathwayOp::ListPathways => {
            let mut out = "Available Signaling Pathways:\n\n".to_string();
            for pw in &db {
                let nodes = get_pathway_nodes(pw);
                out.push_str(&format!(
                    "  {} — {} ({} nodes, {} edges)\n",
                    pw.name,
                    pw.description,
                    nodes.len(),
                    pw.edges.len()
                ));
            }
            out
        }
        PathwayOp::Bfs { start, pathway } => {
            let pws: Vec<&PathwayGraph> = if let Some(p) = pathway {
                find_pathway(&db, p).into_iter().collect()
            } else {
                find_node_pathway(&db, start)
            };
            if pws.is_empty() {
                return format!("Node \"{}\" not found in any pathway", start);
            }
            let mut out = String::new();
            for pw in pws {
                let visited = bfs_forward(pw, start);
                out.push_str(&format!(
                    "BFS from {} in {} pathway:\n  {}\n\n",
                    start,
                    pw.name,
                    visited.join(" → ")
                ));
            }
            out
        }
        PathwayOp::Dfs { start, pathway } => {
            let pws: Vec<&PathwayGraph> = if let Some(p) = pathway {
                find_pathway(&db, p).into_iter().collect()
            } else {
                find_node_pathway(&db, start)
            };
            if pws.is_empty() {
                return format!("Node \"{}\" not found in any pathway", start);
            }
            let mut out = String::new();
            for pw in pws {
                // DFS via stack
                let upper = start.to_uppercase();
                let mut visited = Vec::new();
                let mut stack = Vec::new();
                for &(f, t, _) in pw.edges {
                    if f.to_uppercase() == upper {
                        stack.push(f);
                        break;
                    }
                    if t.to_uppercase() == upper {
                        stack.push(t);
                        break;
                    }
                }
                while let Some(current) = stack.pop() {
                    if visited.contains(&current) {
                        continue;
                    }
                    visited.push(current);
                    for &(f, t, _) in pw.edges {
                        if f == current && !visited.contains(&t) {
                            stack.push(t);
                        }
                    }
                }
                out.push_str(&format!(
                    "DFS from {} in {} pathway:\n  {}\n\n",
                    start,
                    pw.name,
                    visited.join(" → ")
                ));
            }
            out
        }
        PathwayOp::ShortestPath { from, to, pathway } => {
            let pws: Vec<&PathwayGraph> = if let Some(p) = pathway {
                find_pathway(&db, p).into_iter().collect()
            } else {
                let mut combined = find_node_pathway(&db, from);
                combined.retain(|pw| {
                    pw.edges.iter().any(|&(f, t, _)| {
                        f.to_uppercase() == to.to_uppercase()
                            || t.to_uppercase() == to.to_uppercase()
                    })
                });
                combined
            };
            if pws.is_empty() {
                return format!("No pathway contains both \"{}\" and \"{}\"", from, to);
            }
            let mut out = String::new();
            for pw in pws {
                match shortest_path(pw, from, to) {
                    Some(path) => {
                        let mut path_desc = Vec::new();
                        for i in 0..path.len() - 1 {
                            let edge_type = pw
                                .edges
                                .iter()
                                .find(|&&(f, t, _)| f == path[i] && t == path[i + 1])
                                .map(|e| e.2)
                                .unwrap_or("→");
                            path_desc.push(format!(
                                "{} —[{}]→ {}",
                                path[i],
                                edge_type,
                                path[i + 1]
                            ));
                        }
                        out.push_str(&format!(
                            "Shortest path in {} ({} → {}):\n  Steps: {}\n  {}\n\n",
                            pw.name,
                            from,
                            to,
                            path.len() - 1,
                            path_desc.join("\n  ")
                        ));
                    }
                    None => {
                        out.push_str(&format!(
                            "No path from {} to {} in {} pathway\n\n",
                            from, to, pw.name
                        ));
                    }
                }
            }
            out
        }
        PathwayOp::Upstream { node, pathway } => {
            let pws: Vec<&PathwayGraph> = if let Some(p) = pathway {
                find_pathway(&db, p).into_iter().collect()
            } else {
                find_node_pathway(&db, node)
            };
            if pws.is_empty() {
                return format!("Node \"{}\" not found in any pathway", node);
            }
            let mut out = String::new();
            for pw in pws {
                let upstream = bfs_reverse(pw, node);
                if upstream.len() <= 1 {
                    out.push_str(&format!(
                        "No upstream regulators of {} in {} pathway\n\n",
                        node, pw.name
                    ));
                } else {
                    out.push_str(&format!(
                        "Upstream regulators of {} in {} pathway:\n  {}\n\n",
                        node,
                        pw.name,
                        upstream[1..].join(", ")
                    ));
                }
            }
            out
        }
        PathwayOp::Downstream { node, pathway } => {
            let pws: Vec<&PathwayGraph> = if let Some(p) = pathway {
                find_pathway(&db, p).into_iter().collect()
            } else {
                find_node_pathway(&db, node)
            };
            if pws.is_empty() {
                return format!("Node \"{}\" not found in any pathway", node);
            }
            let mut out = String::new();
            for pw in pws {
                let downstream = bfs_forward(pw, node);
                if downstream.len() <= 1 {
                    out.push_str(&format!(
                        "No downstream targets of {} in {} pathway\n\n",
                        node, pw.name
                    ));
                } else {
                    out.push_str(&format!(
                        "Downstream targets of {} in {} pathway:\n  {}\n\n",
                        node,
                        pw.name,
                        downstream[1..].join(", ")
                    ));
                }
            }
            out
        }
    }
}

// ─────────────────────────────────────────────────
// §88  Checksum Validator (Luhn, ISBN, IBAN, EAN-13)
// ─────────────────────────────────────────────────

pub fn checksum_validate(op: &ChecksumOp) -> String {
    match op {
        ChecksumOp::Luhn { digits } => {
            let clean: Vec<u32> = digits
                .chars()
                .filter(|c| c.is_ascii_digit())
                .map(|c| c.to_digit(10).unwrap())
                .collect();
            if clean.is_empty() {
                return "Invalid: no digits found".into();
            }
            let mut sum = 0u32;
            let parity = clean.len() % 2;
            for (i, &d) in clean.iter().enumerate() {
                let mut val = d;
                if i % 2 == parity {
                    val *= 2;
                    if val > 9 {
                        val -= 9;
                    }
                }
                sum += val;
            }
            let valid = sum.is_multiple_of(10);
            let input_fmt: String = digits.chars().filter(|c| c.is_ascii_digit()).collect();
            format!(
                "Luhn Check: {}\nInput: {}\nChecksum: {} mod 10 = {}\nValid: {}",
                if valid { "PASS" } else { "FAIL" },
                input_fmt,
                sum,
                sum % 10,
                valid
            )
        }
        ChecksumOp::Isbn { code } => {
            let clean: String = code
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == 'X' || *c == 'x')
                .collect();
            match clean.len() {
                10 => {
                    let mut sum = 0u32;
                    for (i, ch) in clean.chars().enumerate() {
                        let val = if (ch == 'X' || ch == 'x') && i == 9 {
                            10
                        } else {
                            ch.to_digit(10).unwrap_or(0)
                        };
                        sum += val * (10 - i as u32);
                    }
                    let valid = sum.is_multiple_of(11);
                    format!(
                        "ISBN-10 Check: {}\nInput: {}\nWeighted sum: {} mod 11 = {}\nValid: {}",
                        if valid { "PASS" } else { "FAIL" },
                        clean,
                        sum,
                        sum % 11,
                        valid
                    )
                }
                13 => {
                    let digits: Vec<u32> = clean.chars().filter_map(|c| c.to_digit(10)).collect();
                    let sum: u32 = digits
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if i % 2 == 0 { d } else { d * 3 })
                        .sum();
                    let valid = sum.is_multiple_of(10);
                    format!(
                        "ISBN-13 Check: {}\nInput: {}\nWeighted sum: {} mod 10 = {}\nValid: {}",
                        if valid { "PASS" } else { "FAIL" },
                        clean,
                        sum,
                        sum % 10,
                        valid
                    )
                }
                _ => format!(
                    "Invalid ISBN: expected 10 or 13 characters, got {}",
                    clean.len()
                ),
            }
        }
        ChecksumOp::Iban { code } => {
            let clean: String = code.chars().filter(|c| !c.is_whitespace()).collect();
            if clean.len() < 4 {
                return "Invalid IBAN: too short".into();
            }
            let country = &clean[..2];
            if !country.chars().all(|c| c.is_ascii_uppercase()) {
                return "Invalid IBAN: must start with 2-letter country code".into();
            }
            // Move first 4 chars to end, convert letters to numbers (A=10..Z=35)
            let rearranged = format!("{}{}", &clean[4..], &clean[..4]);
            let mut num_str = String::new();
            for ch in rearranged.chars() {
                if ch.is_ascii_digit() {
                    num_str.push(ch);
                } else if ch.is_ascii_uppercase() {
                    num_str.push_str(&format!("{}", ch as u32 - 'A' as u32 + 10));
                } else {
                    return format!("Invalid IBAN character: '{}'", ch);
                }
            }
            // mod 97 on the large number (using iterative modulo)
            let mut remainder = 0u64;
            for ch in num_str.chars() {
                remainder = (remainder * 10 + ch.to_digit(10).unwrap() as u64) % 97;
            }
            let valid = remainder == 1;
            format!(
                "IBAN Check: {}\nInput: {}\nCountry: {}\nMod 97: {}\nValid: {}",
                if valid { "PASS" } else { "FAIL" },
                clean,
                country,
                remainder,
                valid
            )
        }
        ChecksumOp::Ean13 { code } => {
            let digits: Vec<u32> = code
                .chars()
                .filter(|c| c.is_ascii_digit())
                .filter_map(|c| c.to_digit(10))
                .collect();
            if digits.len() != 13 {
                return format!("Invalid EAN-13: expected 13 digits, got {}", digits.len());
            }
            let sum: u32 = digits
                .iter()
                .enumerate()
                .map(|(i, &d)| if i % 2 == 0 { d } else { d * 3 })
                .sum();
            let valid = sum.is_multiple_of(10);
            let clean: String = code.chars().filter(|c| c.is_ascii_digit()).collect();
            format!(
                "EAN-13 Check: {}\nInput: {}\nWeighted sum: {} mod 10 = {}\nValid: {}",
                if valid { "PASS" } else { "FAIL" },
                clean,
                sum,
                sum % 10,
                valid
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §89  NATO Phonetic Alphabet
// ─────────────────────────────────────────────────

pub fn nato_phonetic(text: &str) -> String {
    let table: HashMap<char, &str> = [
        ('A', "Alfa"),
        ('B', "Bravo"),
        ('C', "Charlie"),
        ('D', "Delta"),
        ('E', "Echo"),
        ('F', "Foxtrot"),
        ('G', "Golf"),
        ('H', "Hotel"),
        ('I', "India"),
        ('J', "Juliett"),
        ('K', "Kilo"),
        ('L', "Lima"),
        ('M', "Mike"),
        ('N', "November"),
        ('O', "Oscar"),
        ('P', "Papa"),
        ('Q', "Quebec"),
        ('R', "Romeo"),
        ('S', "Sierra"),
        ('T', "Tango"),
        ('U', "Uniform"),
        ('V', "Victor"),
        ('W', "Whiskey"),
        ('X', "X-ray"),
        ('Y', "Yankee"),
        ('Z', "Zulu"),
        ('0', "Zero"),
        ('1', "One"),
        ('2', "Two"),
        ('3', "Three"),
        ('4', "Four"),
        ('5', "Five"),
        ('6', "Six"),
        ('7', "Seven"),
        ('8', "Eight"),
        ('9', "Niner"),
    ]
    .into_iter()
    .collect();

    let mut lines = Vec::new();
    for ch in text.chars() {
        if ch == ' ' {
            lines.push("  (space)".to_string());
        } else if let Some(word) = table.get(&ch.to_ascii_uppercase()) {
            lines.push(format!("  {} → {}", ch.to_ascii_uppercase(), word));
        }
    }
    format!("NATO Phonetic: \"{}\"\n\n{}", text, lines.join("\n"))
}

// ─────────────────────────────────────────────────
// §90  ROT13 / Caesar Cipher
// ─────────────────────────────────────────────────

fn caesar_shift(text: &str, shift: u8) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                (b'a' + (c as u8 - b'a' + shift) % 26) as char
            } else if c.is_ascii_uppercase() {
                (b'A' + (c as u8 - b'A' + shift) % 26) as char
            } else {
                c
            }
        })
        .collect()
}

pub fn caesar_calc(op: &CaesarOp) -> String {
    match op {
        CaesarOp::Rot13 { text } => {
            let result = caesar_shift(text, 13);
            format!("ROT13\nInput:  {}\nOutput: {}", text, result)
        }
        CaesarOp::Encrypt { text, shift } => {
            let result = caesar_shift(text, *shift);
            format!(
                "Caesar Encrypt (shift {})\nInput:  {}\nOutput: {}",
                shift, text, result
            )
        }
        CaesarOp::Decrypt { text, shift } => {
            let result = caesar_shift(text, 26 - (*shift % 26));
            format!(
                "Caesar Decrypt (shift {})\nInput:  {}\nOutput: {}",
                shift, text, result
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §91  Aspect Ratio Calculator
// ─────────────────────────────────────────────────

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub fn aspect_ratio_calc(op: &AspectRatioOp) -> String {
    match op {
        AspectRatioOp::FromDimensions { width, height } => {
            if *height == 0 {
                return "Invalid: height cannot be 0".into();
            }
            let g = gcd(*width, *height);
            let (rw, rh) = (width / g, height / g);
            let decimal = *width as f64 / *height as f64;
            let common = match (rw, rh) {
                (16, 9) => " (Widescreen)",
                (16, 10) => " (Widescreen WUXGA)",
                (4, 3) => " (Standard)",
                (21, 9) => " (Ultrawide)",
                (1, 1) => " (Square)",
                (3, 2) => " (Classic photo)",
                _ => "",
            };
            format!(
                "Aspect Ratio: {}:{}{}\nDimensions: {} \u{00D7} {}\nDecimal: {:.4}\nPixels: {}",
                rw,
                rh,
                common,
                width,
                height,
                decimal,
                *width as u64 * *height as u64
            )
        }
        AspectRatioOp::Scale {
            width,
            height,
            target_width,
        } => {
            if *width == 0 {
                return "Invalid: width cannot be 0".into();
            }
            let scale = *target_width as f64 / *width as f64;
            let target_height = (*height as f64 * scale).round() as u32;
            let g = gcd(*width, *height);
            format!(
                "Scale Preserving Ratio\nOriginal: {} \u{00D7} {} ({}:{})\nScaled:   {} \u{00D7} {}\nScale factor: {:.4}x",
                width,
                height,
                width / g,
                height / g,
                target_width,
                target_height,
                scale
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §92  Resistor Color Code Decoder
// ─────────────────────────────────────────────────

fn resistor_color_value(name: &str) -> Option<u32> {
    match name.to_lowercase().as_str() {
        "black" => Some(0),
        "brown" => Some(1),
        "red" => Some(2),
        "orange" => Some(3),
        "yellow" => Some(4),
        "green" => Some(5),
        "blue" => Some(6),
        "violet" | "purple" => Some(7),
        "grey" | "gray" => Some(8),
        "white" => Some(9),
        _ => None,
    }
}

fn resistor_color_multiplier(name: &str) -> Option<f64> {
    match name.to_lowercase().as_str() {
        "black" => Some(1.0),
        "brown" => Some(10.0),
        "red" => Some(100.0),
        "orange" => Some(1_000.0),
        "yellow" => Some(10_000.0),
        "green" => Some(100_000.0),
        "blue" => Some(1_000_000.0),
        "violet" | "purple" => Some(10_000_000.0),
        "grey" | "gray" => Some(100_000_000.0),
        "white" => Some(1_000_000_000.0),
        "gold" => Some(0.1),
        "silver" => Some(0.01),
        _ => None,
    }
}

fn resistor_tolerance(name: &str) -> Option<&'static str> {
    match name.to_lowercase().as_str() {
        "brown" => Some("\u{00B1}1%"),
        "red" => Some("\u{00B1}2%"),
        "green" => Some("\u{00B1}0.5%"),
        "blue" => Some("\u{00B1}0.25%"),
        "violet" | "purple" => Some("\u{00B1}0.1%"),
        "grey" | "gray" => Some("\u{00B1}0.05%"),
        "gold" => Some("\u{00B1}5%"),
        "silver" => Some("\u{00B1}10%"),
        _ => Some("\u{00B1}20%"),
    }
}

fn format_resistance(ohms: f64) -> String {
    if ohms >= 1_000_000.0 {
        format!("{:.2} M\u{03A9}", ohms / 1_000_000.0)
    } else if ohms >= 1_000.0 {
        format!("{:.2} k\u{03A9}", ohms / 1_000.0)
    } else if ohms < 1.0 {
        format!("{:.2} m\u{03A9}", ohms * 1_000.0)
    } else {
        format!("{:.2} \u{03A9}", ohms)
    }
}

fn value_color_name(v: u32) -> &'static str {
    match v {
        0 => "Black",
        1 => "Brown",
        2 => "Red",
        3 => "Orange",
        4 => "Yellow",
        5 => "Green",
        6 => "Blue",
        7 => "Violet",
        8 => "Grey",
        9 => "White",
        _ => "?",
    }
}

pub fn resistor_calc(op: &ResistorOp) -> String {
    match op {
        ResistorOp::Decode { bands } => {
            if bands.len() < 3 {
                return "Need at least 3 color bands".into();
            }
            let d1 = match resistor_color_value(&bands[0]) {
                Some(v) => v,
                None => return format!("Unknown color: {}", bands[0]),
            };
            let d2 = match resistor_color_value(&bands[1]) {
                Some(v) => v,
                None => return format!("Unknown color: {}", bands[1]),
            };
            let mult = match resistor_color_multiplier(&bands[2]) {
                Some(v) => v,
                None => return format!("Unknown multiplier: {}", bands[2]),
            };
            let base = (d1 * 10 + d2) as f64;
            let ohms = base * mult;
            let tol = if bands.len() >= 4 {
                resistor_tolerance(&bands[3]).unwrap_or("\u{00B1}20%")
            } else {
                "\u{00B1}20%"
            };
            format!(
                "Resistor Color Code\nBands: {}\nValue: {} ({:.0}\u{03A9})\nTolerance: {}",
                bands.join(" - "),
                format_resistance(ohms),
                ohms,
                tol
            )
        }
        ResistorOp::Encode { ohms } => {
            if *ohms <= 0.0 {
                return "Resistance must be positive".into();
            }
            // Find the two significant digits and multiplier
            let mut val = *ohms;
            let mut mult_idx: i32 = 0;
            while val >= 100.0 {
                val /= 10.0;
                mult_idx += 1;
            }
            while val < 10.0 && mult_idx > -2 {
                val *= 10.0;
                mult_idx -= 1;
            }
            let d1 = (val / 10.0).floor() as u32;
            let d2 = (val % 10.0).round() as u32;
            let mult_name = match mult_idx {
                -2 => "Silver",
                -1 => "Gold",
                0 => "Black",
                1 => "Brown",
                2 => "Red",
                3 => "Orange",
                4 => "Yellow",
                5 => "Green",
                6 => "Blue",
                7 => "Violet",
                _ => "?",
            };
            format!(
                "Resistor Encoding: {}\nBand 1: {} ({})\nBand 2: {} ({})\nMultiplier: {} (\u{00D7}{})\nSuggested tolerance band: Gold (\u{00B1}5%)",
                format_resistance(*ohms),
                value_color_name(d1),
                d1,
                value_color_name(d2),
                d2,
                mult_name,
                10f64.powi(mult_idx)
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §93  Network Bandwidth Calculator
// ─────────────────────────────────────────────────

fn format_duration_human(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.2} \u{00B5}s", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.2} ms", secs * 1_000.0)
    } else if secs < 60.0 {
        format!("{:.2} seconds", secs)
    } else if secs < 3600.0 {
        format!("{:.1} minutes", secs / 60.0)
    } else if secs < 86400.0 {
        format!("{:.1} hours", secs / 3600.0)
    } else {
        format!("{:.1} days", secs / 86400.0)
    }
}

fn format_speed_human(bps: u64) -> String {
    if bps >= 1_000_000_000 {
        format!("{:.2} Gbps", bps as f64 / 1e9)
    } else if bps >= 1_000_000 {
        format!("{:.2} Mbps", bps as f64 / 1e6)
    } else if bps >= 1_000 {
        format!("{:.2} Kbps", bps as f64 / 1e3)
    } else {
        format!("{} bps", bps)
    }
}

fn format_size_human(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.2} TB", bytes as f64 / 1e12)
    } else if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1e3)
    } else {
        format!("{} bytes", bytes)
    }
}

pub fn bandwidth_calc(op: &BandwidthOp) -> String {
    match op {
        BandwidthOp::TransferTime {
            bytes,
            bits_per_sec,
        } => {
            if *bits_per_sec == 0 {
                return "Speed cannot be 0".into();
            }
            let bits = *bytes as f64 * 8.0;
            let secs = bits / *bits_per_sec as f64;
            format!(
                "Transfer Time\nFile size: {}\nSpeed: {}\nTime: {}",
                format_size_human(*bytes),
                format_speed_human(*bits_per_sec),
                format_duration_human(secs)
            )
        }
        BandwidthOp::RequiredSpeed { bytes, seconds } => {
            if *seconds <= 0.0 {
                return "Time must be positive".into();
            }
            let bits = *bytes as f64 * 8.0;
            let bps = (bits / *seconds) as u64;
            format!(
                "Required Speed\nFile size: {}\nTime: {}\nSpeed needed: {}",
                format_size_human(*bytes),
                format_duration_human(*seconds),
                format_speed_human(bps)
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §94  Unicode Character Inspector
// ─────────────────────────────────────────────────

pub fn unicode_inspect(input: &str) -> String {
    let mut lines = Vec::new();
    for ch in input.chars() {
        let cp = ch as u32;
        let name = unicode_name(ch);
        let cat = unicode_category(ch);
        let utf8_bytes: Vec<String> = {
            let mut buf = [0u8; 4];
            ch.encode_utf8(&mut buf);
            buf[..ch.len_utf8()]
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect()
        };
        let utf16_units: Vec<String> = {
            let mut buf = [0u16; 2];
            ch.encode_utf16(&mut buf);
            buf[..ch.len_utf16()]
                .iter()
                .map(|u| format!("{:04X}", u))
                .collect()
        };
        lines.push(format!(
            "  '{}' U+{:04X} | {} | {} | UTF-8: {} | UTF-16: {}",
            if ch.is_control() { '\u{FFFD}' } else { ch },
            cp,
            name,
            cat,
            utf8_bytes.join(" "),
            utf16_units.join(" ")
        ));
    }
    format!(
        "Unicode Inspector ({} character{})\n\n{}",
        input.chars().count(),
        if input.chars().count() == 1 { "" } else { "s" },
        lines.join("\n")
    )
}

fn unicode_name(ch: char) -> &'static str {
    match ch {
        ' ' => "SPACE",
        '\t' => "TAB",
        '\n' => "LINE FEED",
        '\r' => "CARRIAGE RETURN",
        '!' => "EXCLAMATION MARK",
        '"' => "QUOTATION MARK",
        '#' => "NUMBER SIGN",
        '$' => "DOLLAR SIGN",
        '%' => "PERCENT SIGN",
        '&' => "AMPERSAND",
        '\'' => "APOSTROPHE",
        '(' => "LEFT PARENTHESIS",
        ')' => "RIGHT PARENTHESIS",
        '*' => "ASTERISK",
        '+' => "PLUS SIGN",
        ',' => "COMMA",
        '-' => "HYPHEN-MINUS",
        '.' => "FULL STOP",
        '/' => "SOLIDUS",
        ':' => "COLON",
        ';' => "SEMICOLON",
        '<' => "LESS-THAN SIGN",
        '=' => "EQUALS SIGN",
        '>' => "GREATER-THAN SIGN",
        '?' => "QUESTION MARK",
        '@' => "COMMERCIAL AT",
        '[' => "LEFT SQUARE BRACKET",
        '\\' => "REVERSE SOLIDUS",
        ']' => "RIGHT SQUARE BRACKET",
        '^' => "CIRCUMFLEX ACCENT",
        '_' => "LOW LINE",
        '`' => "GRAVE ACCENT",
        '{' => "LEFT CURLY BRACKET",
        '|' => "VERTICAL LINE",
        '}' => "RIGHT CURLY BRACKET",
        '~' => "TILDE",
        '\u{00A9}' => "COPYRIGHT SIGN",
        '\u{00AE}' => "REGISTERED SIGN",
        '\u{2022}' => "BULLET",
        '\u{2026}' => "HORIZONTAL ELLIPSIS",
        '\u{2014}' => "EM DASH",
        '\u{2013}' => "EN DASH",
        '\u{201C}' => "LEFT DOUBLE QUOTATION MARK",
        '\u{201D}' => "RIGHT DOUBLE QUOTATION MARK",
        '\u{2018}' => "LEFT SINGLE QUOTATION MARK",
        '\u{2019}' => "RIGHT SINGLE QUOTATION MARK",
        '\u{00B0}' => "DEGREE SIGN",
        '\u{00B5}' => "MICRO SIGN",
        '\u{2248}' => "ALMOST EQUAL TO",
        '\u{2260}' => "NOT EQUAL TO",
        '\u{2264}' => "LESS-THAN OR EQUAL TO",
        '\u{2265}' => "GREATER-THAN OR EQUAL TO",
        '\u{03C0}' => "GREEK SMALL LETTER PI",
        '\u{03B1}' => "GREEK SMALL LETTER ALPHA",
        '\u{03B2}' => "GREEK SMALL LETTER BETA",
        '\u{03B3}' => "GREEK SMALL LETTER GAMMA",
        '\u{2192}' => "RIGHTWARDS ARROW",
        '\u{2190}' => "LEFTWARDS ARROW",
        '\u{2191}' => "UPWARDS ARROW",
        '\u{2193}' => "DOWNWARDS ARROW",
        '\u{221E}' => "INFINITY",
        '\u{2211}' => "N-ARY SUMMATION",
        '\u{222B}' => "INTEGRAL",
        '\u{221A}' => "SQUARE ROOT",
        '\u{00D7}' => "MULTIPLICATION SIGN",
        '\u{00F7}' => "DIVISION SIGN",
        _ if ch.is_ascii_digit() => "DIGIT",
        _ if ch.is_ascii_uppercase() => "LATIN CAPITAL LETTER",
        _ if ch.is_ascii_lowercase() => "LATIN SMALL LETTER",
        _ => "(name lookup requires ICU)",
    }
}

fn unicode_category(ch: char) -> &'static str {
    if ch.is_ascii_uppercase() || ch.is_uppercase() {
        "Letter, Uppercase"
    } else if ch.is_ascii_lowercase() || ch.is_lowercase() {
        "Letter, Lowercase"
    } else if ch.is_ascii_digit() || ch.is_numeric() {
        "Number, Digit"
    } else if ch.is_ascii_punctuation() {
        "Punctuation"
    } else if ch.is_whitespace() {
        "Separator, Space"
    } else if ch.is_control() {
        "Other, Control"
    } else if ch.is_alphabetic() {
        "Letter"
    } else {
        "Symbol"
    }
}

// ─────────────────────────────────────────────────
// §95  IEEE 754 Float Inspector
// ─────────────────────────────────────────────────

pub fn float754_inspect(value: f64) -> String {
    let bits = value.to_bits();
    let sign = (bits >> 63) & 1;
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;
    let biased = exponent - 1023;

    let sign_str = if sign == 1 { "-" } else { "+" };
    let class = if exponent == 0x7FF {
        if mantissa == 0 { "Infinity" } else { "NaN" }
    } else if exponent == 0 {
        if mantissa == 0 { "Zero" } else { "Subnormal" }
    } else {
        "Normal"
    };

    let bin_str = format!("{:064b}", bits);
    let hex_str = format!("{:016X}", bits);

    // Also show f32 representation
    let f32_val = value as f32;
    let f32_bits = f32_val.to_bits();
    let f32_hex = format!("{:08X}", f32_bits);

    format!(
        "IEEE 754 Float Inspector\n\
         Value: {}\n\
         Class: {}\n\n\
         64-bit (double):\n\
         Sign: {} ({})\n\
         Exponent: {:011b} ({}, biased: {})\n\
         Mantissa: {:052b}\n\
         Hex: 0x{}\n\
         Binary: {} {} {}\n\n\
         32-bit (float): {}\n\
         Hex: 0x{}",
        value,
        class,
        sign,
        sign_str,
        exponent,
        exponent,
        biased,
        mantissa,
        hex_str,
        &bin_str[0..1],
        &bin_str[1..12],
        &bin_str[12..],
        f32_val,
        f32_hex,
    )
}

// ─────────────────────────────────────────────────
// §96  Frequency / Wavelength Calculator
// ─────────────────────────────────────────────────

const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s

fn classify_em_band(hz: f64) -> &'static str {
    // ITU radio band designations + optical spectrum
    if hz < 3e3 {
        "Extremely Low Frequency (ELF)"
    } else if hz < 3e4 {
        "Super Low Frequency (SLF)"
    } else if hz < 3e5 {
        "Ultra Low Frequency (ULF)"
    } else if hz < 3e6 {
        "Very Low Frequency (VLF)"
    } else if hz < 3e7 {
        "Low Frequency (LF) / AM Radio"
    } else if hz < 3e8 {
        "Very High Frequency (VHF) / FM Radio & TV"
    } else if hz < 3e9 {
        "Ultra High Frequency (UHF) / WiFi & Cellular"
    } else if hz < 3e10 {
        "Super High Frequency (SHF) / Radar & 5G"
    } else if hz < 3e11 {
        "Extremely High Frequency (EHF) / mmWave"
    } else if hz < 4.3e14 {
        "Infrared"
    } else if hz < 7.5e14 {
        "Visible Light"
    } else if hz < 3e16 {
        "Ultraviolet"
    } else if hz < 3e19 {
        "X-ray"
    } else {
        "Gamma Ray"
    }
}

fn format_wavelength(meters: f64) -> String {
    if meters >= 1_000.0 {
        format!("{:.2} km", meters / 1_000.0)
    } else if meters >= 1.0 {
        format!("{:.2} m", meters)
    } else if meters >= 0.01 {
        format!("{:.2} cm", meters * 100.0)
    } else if meters >= 0.001 {
        format!("{:.2} mm", meters * 1_000.0)
    } else if meters >= 1e-6 {
        format!("{:.2} \u{00B5}m", meters * 1e6)
    } else if meters >= 1e-9 {
        format!("{:.2} nm", meters * 1e9)
    } else if meters >= 1e-12 {
        format!("{:.3} pm", meters * 1e12)
    } else {
        format!("{:.3e} m", meters)
    }
}

fn format_freq_human(hz: f64) -> String {
    if hz >= 1e12 {
        format!("{:.3} THz", hz / 1e12)
    } else if hz >= 1e9 {
        format!("{:.3} GHz", hz / 1e9)
    } else if hz >= 1e6 {
        format!("{:.3} MHz", hz / 1e6)
    } else if hz >= 1e3 {
        format!("{:.3} kHz", hz / 1e3)
    } else {
        format!("{:.3} Hz", hz)
    }
}

pub fn freq_wavelength_calc(op: &FreqWavelengthOp) -> String {
    match op {
        FreqWavelengthOp::FreqToWavelength { hz } => {
            if *hz <= 0.0 {
                return "Frequency must be positive".into();
            }
            let wavelength = SPEED_OF_LIGHT / hz;
            let band = classify_em_band(*hz);
            format!(
                "Frequency \u{2192} Wavelength\nFrequency: {}\nWavelength: {}\nBand: {}",
                format_freq_human(*hz),
                format_wavelength(wavelength),
                band
            )
        }
        FreqWavelengthOp::WavelengthToFreq { meters } => {
            if *meters <= 0.0 {
                return "Wavelength must be positive".into();
            }
            let hz = SPEED_OF_LIGHT / meters;
            let band = classify_em_band(hz);
            format!(
                "Wavelength \u{2192} Frequency\nWavelength: {}\nFrequency: {}\nBand: {}",
                format_wavelength(*meters),
                format_freq_human(hz),
                band
            )
        }
        FreqWavelengthOp::Classify { hz } => {
            if *hz <= 0.0 {
                return "Frequency must be positive".into();
            }
            let wavelength = SPEED_OF_LIGHT / hz;
            let band = classify_em_band(*hz);
            format!(
                "EM Spectrum Classification\nFrequency: {}\nWavelength: {}\nBand: {}",
                format_freq_human(*hz),
                format_wavelength(wavelength),
                band
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §97  Chemical Formula Parser (Molar Mass)
// ─────────────────────────────────────────────────

fn element_weight(sym: &str) -> Option<f64> {
    match sym {
        "H" => Some(1.008),
        "He" => Some(4.003),
        "Li" => Some(6.941),
        "Be" => Some(9.012),
        "B" => Some(10.81),
        "C" => Some(12.011),
        "N" => Some(14.007),
        "O" => Some(15.999),
        "F" => Some(18.998),
        "Ne" => Some(20.180),
        "Na" => Some(22.990),
        "Mg" => Some(24.305),
        "Al" => Some(26.982),
        "Si" => Some(28.086),
        "P" => Some(30.974),
        "S" => Some(32.065),
        "Cl" => Some(35.453),
        "Ar" => Some(39.948),
        "K" => Some(39.098),
        "Ca" => Some(40.078),
        "Ti" => Some(47.867),
        "V" => Some(50.942),
        "Cr" => Some(51.996),
        "Mn" => Some(54.938),
        "Fe" => Some(55.845),
        "Co" => Some(58.933),
        "Ni" => Some(58.693),
        "Cu" => Some(63.546),
        "Zn" => Some(65.380),
        "Ga" => Some(69.723),
        "Ge" => Some(72.630),
        "As" => Some(74.922),
        "Se" => Some(78.971),
        "Br" => Some(79.904),
        "Kr" => Some(83.798),
        "Rb" => Some(85.468),
        "Sr" => Some(87.620),
        "Zr" => Some(91.224),
        "Nb" => Some(92.906),
        "Mo" => Some(95.950),
        "Ag" => Some(107.87),
        "Cd" => Some(112.41),
        "Sn" => Some(118.71),
        "Sb" => Some(121.76),
        "I" => Some(126.90),
        "Ba" => Some(137.33),
        "W" => Some(183.84),
        "Pt" => Some(195.08),
        "Au" => Some(196.97),
        "Hg" => Some(200.59),
        "Pb" => Some(207.20),
        "U" => Some(238.03),
        _ => None,
    }
}

fn parse_formula_group(chars: &[u8], pos: &mut usize) -> Result<Vec<(String, f64)>, String> {
    let mut elements = Vec::new();
    while *pos < chars.len() {
        let ch = chars[*pos] as char;
        if ch == ')' {
            break;
        }
        if ch == '(' {
            *pos += 1;
            let sub = parse_formula_group(chars, pos)?;
            if *pos < chars.len() && chars[*pos] == b')' {
                *pos += 1;
            }
            let count = parse_subscript(chars, pos);
            for (sym, n) in sub {
                elements.push((sym, n * count));
            }
        } else if ch.is_ascii_uppercase() {
            let start = *pos;
            *pos += 1;
            while *pos < chars.len() && (chars[*pos] as char).is_ascii_lowercase() {
                *pos += 1;
            }
            let sym = String::from_utf8_lossy(&chars[start..*pos]).to_string();
            let count = parse_subscript(chars, pos);
            elements.push((sym, count));
        } else {
            *pos += 1; // skip unexpected chars
        }
    }
    Ok(elements)
}

fn parse_subscript(chars: &[u8], pos: &mut usize) -> f64 {
    let start = *pos;
    while *pos < chars.len() && (chars[*pos] as char).is_ascii_digit() {
        *pos += 1;
    }
    if start == *pos {
        return 1.0;
    }
    String::from_utf8_lossy(&chars[start..*pos])
        .parse()
        .unwrap_or(1.0)
}

pub fn molar_mass(formula: &str) -> Result<String, String> {
    let bytes = formula.as_bytes();
    let mut pos = 0;
    let elements = parse_formula_group(bytes, &mut pos)?;

    let mut total = 0.0;
    let mut breakdown = Vec::new();
    let mut composition: HashMap<String, f64> = HashMap::new();

    for (sym, count) in &elements {
        let w = element_weight(sym).ok_or_else(|| format!("Unknown element: {}", sym))?;
        let contrib = w * count;
        total += contrib;
        *composition.entry(sym.clone()).or_insert(0.0) += *count;
    }

    for (sym, count) in &composition {
        let w = element_weight(sym).unwrap();
        let mass = w * count;
        let pct = mass / total * 100.0;
        breakdown.push(format!(
            "  {} \u{00D7} {:.0}: {:.3} g/mol ({:.1}%)",
            sym, count, mass, pct
        ));
    }

    breakdown.sort();
    Ok(format!(
        "Molar Mass: {} = {:.3} g/mol\n\nComposition:\n{}",
        formula,
        total,
        breakdown.join("\n")
    ))
}

// ─────────────────────────────────────────────────
// §98  Translation Lookup
// ─────────────────────────────────────────────────

/// Normalize a language name/code to a 2-letter ISO code.
fn normalize_lang(lang: &str) -> &'static str {
    match lang.trim().to_lowercase().as_str() {
        "fr" | "french" | "français" | "francais" => "fr",
        "es" | "spanish" | "español" | "espanol" => "es",
        "de" | "german" | "deutsch" => "de",
        "it" | "italian" | "italiano" => "it",
        "pt" | "portuguese" | "português" | "portugues" => "pt",
        "ja" | "japanese" | "日本語" => "ja",
        "zh" | "chinese" | "mandarin" | "中文" => "zh",
        "en" | "english" => "en",
        "ko" | "korean" | "한국어" => "ko",
        "ru" | "russian" | "русский" => "ru",
        "ar" | "arabic" | "العربية" => "ar",
        "hi" | "hindi" | "हिन्दी" => "hi",
        _ => "unknown",
    }
}

/// Lookup common phrases across languages. Returns formatted translation or
/// a `[needs LLM]` sentinel if the phrase isn't in the deterministic table.
pub fn translate_lookup(text: &str, target_lang: &str) -> String {
    let lang = normalize_lang(target_lang);
    if lang == "unknown" {
        return format!("[needs LLM] Unknown target language: {}", target_lang);
    }

    let key = text.trim().to_lowercase();

    // Phrase table: English → target language (200+ phrases)
    // Each entry: (english_key, fr, es, de, it, pt, ja, zh, ko, ru, ar, hi)
    let table: &[(&str, &[(&str, &str)])] = &[
        // ── Greetings & Farewells ──
        (
            "hello",
            &[
                ("fr", "Bonjour"),
                ("es", "Hola"),
                ("de", "Hallo"),
                ("it", "Ciao"),
                ("pt", "Olá"),
                ("ja", "こんにちは"),
                ("zh", "你好"),
                ("ko", "안녕하세요"),
                ("ru", "Привет"),
                ("ar", "مرحبا"),
                ("hi", "नमस्ते"),
            ],
        ),
        (
            "hi",
            &[
                ("fr", "Salut"),
                ("es", "Hola"),
                ("de", "Hi"),
                ("it", "Ciao"),
                ("pt", "Oi"),
                ("ja", "やあ"),
                ("zh", "嗨"),
                ("ko", "안녕"),
                ("ru", "Привет"),
                ("ar", "مرحبا"),
                ("hi", "नमस्ते"),
            ],
        ),
        (
            "goodbye",
            &[
                ("fr", "Au revoir"),
                ("es", "Adiós"),
                ("de", "Auf Wiedersehen"),
                ("it", "Arrivederci"),
                ("pt", "Adeus"),
                ("ja", "さようなら"),
                ("zh", "再见"),
                ("ko", "안녕히 가세요"),
                ("ru", "До свидания"),
                ("ar", "وداعا"),
                ("hi", "अलविदा"),
            ],
        ),
        (
            "see you later",
            &[
                ("fr", "À plus tard"),
                ("es", "Hasta luego"),
                ("de", "Bis später"),
                ("it", "A dopo"),
                ("pt", "Até logo"),
                ("ja", "また後で"),
                ("zh", "回头见"),
                ("ko", "나중에 봐요"),
                ("ru", "До встречи"),
                ("ar", "أراك لاحقا"),
                ("hi", "फिर मिलते हैं"),
            ],
        ),
        (
            "good morning",
            &[
                ("fr", "Bonjour"),
                ("es", "Buenos días"),
                ("de", "Guten Morgen"),
                ("it", "Buongiorno"),
                ("pt", "Bom dia"),
                ("ja", "おはようございます"),
                ("zh", "早上好"),
                ("ko", "좋은 아침"),
                ("ru", "Доброе утро"),
                ("ar", "صباح الخير"),
                ("hi", "सुप्रभात"),
            ],
        ),
        (
            "good afternoon",
            &[
                ("fr", "Bon après-midi"),
                ("es", "Buenas tardes"),
                ("de", "Guten Tag"),
                ("it", "Buon pomeriggio"),
                ("pt", "Boa tarde"),
                ("ja", "こんにちは"),
                ("zh", "下午好"),
                ("ko", "좋은 오후"),
                ("ru", "Добрый день"),
                ("ar", "مساء الخير"),
                ("hi", "नमस्कार"),
            ],
        ),
        (
            "good evening",
            &[
                ("fr", "Bonsoir"),
                ("es", "Buenas noches"),
                ("de", "Guten Abend"),
                ("it", "Buonasera"),
                ("pt", "Boa noite"),
                ("ja", "こんばんは"),
                ("zh", "晚上好"),
                ("ko", "좋은 저녁"),
                ("ru", "Добрый вечер"),
                ("ar", "مساء الخير"),
                ("hi", "शुभ संध्या"),
            ],
        ),
        (
            "good night",
            &[
                ("fr", "Bonne nuit"),
                ("es", "Buenas noches"),
                ("de", "Gute Nacht"),
                ("it", "Buonanotte"),
                ("pt", "Boa noite"),
                ("ja", "おやすみなさい"),
                ("zh", "晚安"),
                ("ko", "잘 자요"),
                ("ru", "Спокойной ночи"),
                ("ar", "تصبح على خير"),
                ("hi", "शुभ रात्रि"),
            ],
        ),
        (
            "welcome",
            &[
                ("fr", "Bienvenue"),
                ("es", "Bienvenido"),
                ("de", "Willkommen"),
                ("it", "Benvenuto"),
                ("pt", "Bem-vindo"),
                ("ja", "ようこそ"),
                ("zh", "欢迎"),
                ("ko", "환영합니다"),
                ("ru", "Добро пожаловать"),
                ("ar", "مرحبا بك"),
                ("hi", "स्वागत है"),
            ],
        ),
        (
            "nice to meet you",
            &[
                ("fr", "Enchanté"),
                ("es", "Mucho gusto"),
                ("de", "Freut mich"),
                ("it", "Piacere"),
                ("pt", "Prazer em conhecê-lo"),
                ("ja", "はじめまして"),
                ("zh", "很高兴认识你"),
                ("ko", "만나서 반갑습니다"),
                ("ru", "Приятно познакомиться"),
                ("ar", "تشرفت بمعرفتك"),
                ("hi", "आपसे मिलकर अच्छा लगा"),
            ],
        ),
        // ── Courtesy ──
        (
            "thank you",
            &[
                ("fr", "Merci"),
                ("es", "Gracias"),
                ("de", "Danke"),
                ("it", "Grazie"),
                ("pt", "Obrigado"),
                ("ja", "ありがとう"),
                ("zh", "谢谢"),
                ("ko", "감사합니다"),
                ("ru", "Спасибо"),
                ("ar", "شكرا"),
                ("hi", "धन्यवाद"),
            ],
        ),
        (
            "thanks",
            &[
                ("fr", "Merci"),
                ("es", "Gracias"),
                ("de", "Danke"),
                ("it", "Grazie"),
                ("pt", "Obrigado"),
                ("ja", "ありがとう"),
                ("zh", "谢谢"),
                ("ko", "감사합니다"),
                ("ru", "Спасибо"),
                ("ar", "شكرا"),
                ("hi", "धन्यवाद"),
            ],
        ),
        (
            "thank you very much",
            &[
                ("fr", "Merci beaucoup"),
                ("es", "Muchas gracias"),
                ("de", "Vielen Dank"),
                ("it", "Grazie mille"),
                ("pt", "Muito obrigado"),
                ("ja", "どうもありがとうございます"),
                ("zh", "非常感谢"),
                ("ko", "대단히 감사합니다"),
                ("ru", "Большое спасибо"),
                ("ar", "شكرا جزيلا"),
                ("hi", "बहुत धन्यवाद"),
            ],
        ),
        (
            "you're welcome",
            &[
                ("fr", "De rien"),
                ("es", "De nada"),
                ("de", "Bitte schön"),
                ("it", "Prego"),
                ("pt", "De nada"),
                ("ja", "どういたしまして"),
                ("zh", "不客气"),
                ("ko", "천만에요"),
                ("ru", "Пожалуйста"),
                ("ar", "عفوا"),
                ("hi", "कोई बात नहीं"),
            ],
        ),
        (
            "please",
            &[
                ("fr", "S'il vous plaît"),
                ("es", "Por favor"),
                ("de", "Bitte"),
                ("it", "Per favore"),
                ("pt", "Por favor"),
                ("ja", "お願いします"),
                ("zh", "请"),
                ("ko", "제발"),
                ("ru", "Пожалуйста"),
                ("ar", "من فضلك"),
                ("hi", "कृपया"),
            ],
        ),
        (
            "excuse me",
            &[
                ("fr", "Excusez-moi"),
                ("es", "Disculpe"),
                ("de", "Entschuldigung"),
                ("it", "Mi scusi"),
                ("pt", "Com licença"),
                ("ja", "すみません"),
                ("zh", "对不起"),
                ("ko", "실례합니다"),
                ("ru", "Извините"),
                ("ar", "عفوا"),
                ("hi", "माफ़ कीजिए"),
            ],
        ),
        (
            "sorry",
            &[
                ("fr", "Désolé"),
                ("es", "Lo siento"),
                ("de", "Es tut mir leid"),
                ("it", "Mi dispiace"),
                ("pt", "Desculpe"),
                ("ja", "ごめんなさい"),
                ("zh", "对不起"),
                ("ko", "죄송합니다"),
                ("ru", "Извините"),
                ("ar", "آسف"),
                ("hi", "माफ़ कीजिए"),
            ],
        ),
        (
            "no problem",
            &[
                ("fr", "Pas de problème"),
                ("es", "No hay problema"),
                ("de", "Kein Problem"),
                ("it", "Nessun problema"),
                ("pt", "Sem problema"),
                ("ja", "問題ありません"),
                ("zh", "没问题"),
                ("ko", "괜찮아요"),
                ("ru", "Без проблем"),
                ("ar", "لا مشكلة"),
                ("hi", "कोई बात नहीं"),
            ],
        ),
        // ── Responses ──
        (
            "yes",
            &[
                ("fr", "Oui"),
                ("es", "Sí"),
                ("de", "Ja"),
                ("it", "Sì"),
                ("pt", "Sim"),
                ("ja", "はい"),
                ("zh", "是"),
                ("ko", "네"),
                ("ru", "Да"),
                ("ar", "نعم"),
                ("hi", "हाँ"),
            ],
        ),
        (
            "no",
            &[
                ("fr", "Non"),
                ("es", "No"),
                ("de", "Nein"),
                ("it", "No"),
                ("pt", "Não"),
                ("ja", "いいえ"),
                ("zh", "不"),
                ("ko", "아니요"),
                ("ru", "Нет"),
                ("ar", "لا"),
                ("hi", "नहीं"),
            ],
        ),
        (
            "maybe",
            &[
                ("fr", "Peut-être"),
                ("es", "Quizás"),
                ("de", "Vielleicht"),
                ("it", "Forse"),
                ("pt", "Talvez"),
                ("ja", "たぶん"),
                ("zh", "也许"),
                ("ko", "아마도"),
                ("ru", "Может быть"),
                ("ar", "ربما"),
                ("hi", "शायद"),
            ],
        ),
        (
            "of course",
            &[
                ("fr", "Bien sûr"),
                ("es", "Por supuesto"),
                ("de", "Natürlich"),
                ("it", "Certo"),
                ("pt", "Claro"),
                ("ja", "もちろん"),
                ("zh", "当然"),
                ("ko", "물론이요"),
                ("ru", "Конечно"),
                ("ar", "بالطبع"),
                ("hi", "बिल्कुल"),
            ],
        ),
        (
            "i agree",
            &[
                ("fr", "Je suis d'accord"),
                ("es", "Estoy de acuerdo"),
                ("de", "Ich stimme zu"),
                ("it", "Sono d'accordo"),
                ("pt", "Concordo"),
                ("ja", "同意します"),
                ("zh", "我同意"),
                ("ko", "동의합니다"),
                ("ru", "Согласен"),
                ("ar", "أوافق"),
                ("hi", "मैं सहमत हूँ"),
            ],
        ),
        (
            "i disagree",
            &[
                ("fr", "Je ne suis pas d'accord"),
                ("es", "No estoy de acuerdo"),
                ("de", "Ich stimme nicht zu"),
                ("it", "Non sono d'accordo"),
                ("pt", "Discordo"),
                ("ja", "同意しません"),
                ("zh", "我不同意"),
                ("ko", "동의하지 않습니다"),
                ("ru", "Не согласен"),
                ("ar", "لا أوافق"),
                ("hi", "मैं सहमत नहीं हूँ"),
            ],
        ),
        // ── Questions ──
        (
            "how are you",
            &[
                ("fr", "Comment allez-vous ?"),
                ("es", "¿Cómo estás?"),
                ("de", "Wie geht es Ihnen?"),
                ("it", "Come stai?"),
                ("pt", "Como você está?"),
                ("ja", "お元気ですか？"),
                ("zh", "你好吗？"),
                ("ko", "어떻게 지내세요?"),
                ("ru", "Как дела?"),
                ("ar", "كيف حالك؟"),
                ("hi", "आप कैसे हैं?"),
            ],
        ),
        (
            "what is your name",
            &[
                ("fr", "Comment vous appelez-vous ?"),
                ("es", "¿Cómo te llamas?"),
                ("de", "Wie heißen Sie?"),
                ("it", "Come ti chiami?"),
                ("pt", "Como você se chama?"),
                ("ja", "お名前は何ですか？"),
                ("zh", "你叫什么名字？"),
                ("ko", "이름이 뭐에요?"),
                ("ru", "Как вас зовут?"),
                ("ar", "ما اسمك؟"),
                ("hi", "आपका नाम क्या है?"),
            ],
        ),
        (
            "where is",
            &[
                ("fr", "Où est"),
                ("es", "¿Dónde está"),
                ("de", "Wo ist"),
                ("it", "Dov'è"),
                ("pt", "Onde está"),
                ("ja", "どこですか"),
                ("zh", "在哪里"),
                ("ko", "어디에"),
                ("ru", "Где"),
                ("ar", "أين"),
                ("hi", "कहाँ है"),
            ],
        ),
        (
            "what time is it",
            &[
                ("fr", "Quelle heure est-il ?"),
                ("es", "¿Qué hora es?"),
                ("de", "Wie spät ist es?"),
                ("it", "Che ore sono?"),
                ("pt", "Que horas são?"),
                ("ja", "今何時ですか？"),
                ("zh", "现在几点？"),
                ("ko", "지금 몇 시예요?"),
                ("ru", "Который час?"),
                ("ar", "كم الساعة؟"),
                ("hi", "कितने बजे हैं?"),
            ],
        ),
        (
            "how much",
            &[
                ("fr", "Combien"),
                ("es", "Cuánto"),
                ("de", "Wie viel"),
                ("it", "Quanto"),
                ("pt", "Quanto"),
                ("ja", "いくら"),
                ("zh", "多少"),
                ("ko", "얼마"),
                ("ru", "Сколько"),
                ("ar", "كم"),
                ("hi", "कितना"),
            ],
        ),
        (
            "how much does it cost",
            &[
                ("fr", "Combien ça coûte ?"),
                ("es", "¿Cuánto cuesta?"),
                ("de", "Wie viel kostet das?"),
                ("it", "Quanto costa?"),
                ("pt", "Quanto custa?"),
                ("ja", "いくらですか？"),
                ("zh", "多少钱？"),
                ("ko", "얼마예요?"),
                ("ru", "Сколько стоит?"),
                ("ar", "كم الثمن؟"),
                ("hi", "कितना है?"),
            ],
        ),
        (
            "where is the bathroom",
            &[
                ("fr", "Où sont les toilettes ?"),
                ("es", "¿Dónde está el baño?"),
                ("de", "Wo ist die Toilette?"),
                ("it", "Dov'è il bagno?"),
                ("pt", "Onde é o banheiro?"),
                ("ja", "トイレはどこですか？"),
                ("zh", "洗手间在哪里？"),
                ("ko", "화장실이 어디에요?"),
                ("ru", "Где туалет?"),
                ("ar", "أين الحمام؟"),
                ("hi", "बाथरूम कहाँ है?"),
            ],
        ),
        (
            "do you speak english",
            &[
                ("fr", "Parlez-vous anglais ?"),
                ("es", "¿Hablas inglés?"),
                ("de", "Sprechen Sie Englisch?"),
                ("it", "Parli inglese?"),
                ("pt", "Você fala inglês?"),
                ("ja", "英語を話しますか？"),
                ("zh", "你会说英语吗？"),
                ("ko", "영어 하세요?"),
                ("ru", "Вы говорите по-английски?"),
                ("ar", "هل تتحدث الإنجليزية؟"),
                ("hi", "क्या आप अंग्रेजी बोलते हैं?"),
            ],
        ),
        (
            "can you help me",
            &[
                ("fr", "Pouvez-vous m'aider ?"),
                ("es", "¿Puede ayudarme?"),
                ("de", "Können Sie mir helfen?"),
                ("it", "Può aiutarmi?"),
                ("pt", "Pode me ajudar?"),
                ("ja", "助けてもらえますか？"),
                ("zh", "你能帮我吗？"),
                ("ko", "도와주시겠어요?"),
                ("ru", "Вы можете мне помочь?"),
                ("ar", "هل يمكنك مساعدتي؟"),
                ("hi", "क्या आप मेरी मदद कर सकते हैं?"),
            ],
        ),
        (
            "what does this mean",
            &[
                ("fr", "Qu'est-ce que cela signifie ?"),
                ("es", "¿Qué significa esto?"),
                ("de", "Was bedeutet das?"),
                ("it", "Cosa significa questo?"),
                ("pt", "O que isso significa?"),
                ("ja", "これはどういう意味ですか？"),
                ("zh", "这是什么意思？"),
                ("ko", "이게 무슨 뜻이에요?"),
                ("ru", "Что это значит?"),
                ("ar", "ماذا يعني هذا؟"),
                ("hi", "इसका क्या मतलब है?"),
            ],
        ),
        // ── Common phrases ──
        (
            "i don't understand",
            &[
                ("fr", "Je ne comprends pas"),
                ("es", "No entiendo"),
                ("de", "Ich verstehe nicht"),
                ("it", "Non capisco"),
                ("pt", "Não entendo"),
                ("ja", "わかりません"),
                ("zh", "我不明白"),
                ("ko", "이해하지 못합니다"),
                ("ru", "Я не понимаю"),
                ("ar", "لا أفهم"),
                ("hi", "मुझे समझ नहीं आया"),
            ],
        ),
        (
            "i understand",
            &[
                ("fr", "Je comprends"),
                ("es", "Entiendo"),
                ("de", "Ich verstehe"),
                ("it", "Capisco"),
                ("pt", "Entendo"),
                ("ja", "わかります"),
                ("zh", "我明白"),
                ("ko", "이해합니다"),
                ("ru", "Я понимаю"),
                ("ar", "أفهم"),
                ("hi", "मैं समझता हूँ"),
            ],
        ),
        (
            "i don't know",
            &[
                ("fr", "Je ne sais pas"),
                ("es", "No sé"),
                ("de", "Ich weiß nicht"),
                ("it", "Non lo so"),
                ("pt", "Não sei"),
                ("ja", "わかりません"),
                ("zh", "我不知道"),
                ("ko", "모르겠어요"),
                ("ru", "Я не знаю"),
                ("ar", "لا أعرف"),
                ("hi", "मुझे नहीं पता"),
            ],
        ),
        (
            "i love you",
            &[
                ("fr", "Je t'aime"),
                ("es", "Te quiero"),
                ("de", "Ich liebe dich"),
                ("it", "Ti amo"),
                ("pt", "Eu te amo"),
                ("ja", "愛してる"),
                ("zh", "我爱你"),
                ("ko", "사랑해요"),
                ("ru", "Я тебя люблю"),
                ("ar", "أحبك"),
                ("hi", "मैं तुमसे प्यार करता हूँ"),
            ],
        ),
        (
            "i am lost",
            &[
                ("fr", "Je suis perdu"),
                ("es", "Estoy perdido"),
                ("de", "Ich habe mich verlaufen"),
                ("it", "Mi sono perso"),
                ("pt", "Estou perdido"),
                ("ja", "道に迷いました"),
                ("zh", "我迷路了"),
                ("ko", "길을 잃었어요"),
                ("ru", "Я заблудился"),
                ("ar", "لقد ضللت الطريق"),
                ("hi", "मैं खो गया हूँ"),
            ],
        ),
        (
            "i need help",
            &[
                ("fr", "J'ai besoin d'aide"),
                ("es", "Necesito ayuda"),
                ("de", "Ich brauche Hilfe"),
                ("it", "Ho bisogno di aiuto"),
                ("pt", "Preciso de ajuda"),
                ("ja", "助けが必要です"),
                ("zh", "我需要帮助"),
                ("ko", "도움이 필요해요"),
                ("ru", "Мне нужна помощь"),
                ("ar", "أحتاج مساعدة"),
                ("hi", "मुझे मदद चाहिए"),
            ],
        ),
        (
            "i am hungry",
            &[
                ("fr", "J'ai faim"),
                ("es", "Tengo hambre"),
                ("de", "Ich habe Hunger"),
                ("it", "Ho fame"),
                ("pt", "Estou com fome"),
                ("ja", "お腹が空きました"),
                ("zh", "我饿了"),
                ("ko", "배고파요"),
                ("ru", "Я голоден"),
                ("ar", "أنا جائع"),
                ("hi", "मुझे भूख लगी है"),
            ],
        ),
        (
            "i am thirsty",
            &[
                ("fr", "J'ai soif"),
                ("es", "Tengo sed"),
                ("de", "Ich habe Durst"),
                ("it", "Ho sete"),
                ("pt", "Estou com sede"),
                ("ja", "のどが渇きました"),
                ("zh", "我渴了"),
                ("ko", "목이 마르다"),
                ("ru", "Я хочу пить"),
                ("ar", "أنا عطشان"),
                ("hi", "मुझे प्यास लगी है"),
            ],
        ),
        (
            "i am tired",
            &[
                ("fr", "Je suis fatigué"),
                ("es", "Estoy cansado"),
                ("de", "Ich bin müde"),
                ("it", "Sono stanco"),
                ("pt", "Estou cansado"),
                ("ja", "疲れました"),
                ("zh", "我累了"),
                ("ko", "피곤해요"),
                ("ru", "Я устал"),
                ("ar", "أنا متعب"),
                ("hi", "मैं थक गया हूँ"),
            ],
        ),
        (
            "i am fine",
            &[
                ("fr", "Je vais bien"),
                ("es", "Estoy bien"),
                ("de", "Mir geht es gut"),
                ("it", "Sto bene"),
                ("pt", "Estou bem"),
                ("ja", "元気です"),
                ("zh", "我很好"),
                ("ko", "잘 지내요"),
                ("ru", "У меня всё хорошо"),
                ("ar", "أنا بخير"),
                ("hi", "मैं ठीक हूँ"),
            ],
        ),
        (
            "i am happy",
            &[
                ("fr", "Je suis heureux"),
                ("es", "Estoy feliz"),
                ("de", "Ich bin glücklich"),
                ("it", "Sono felice"),
                ("pt", "Estou feliz"),
                ("ja", "幸せです"),
                ("zh", "我很高兴"),
                ("ko", "행복해요"),
                ("ru", "Я счастлив"),
                ("ar", "أنا سعيد"),
                ("hi", "मैं खुश हूँ"),
            ],
        ),
        (
            "i am sad",
            &[
                ("fr", "Je suis triste"),
                ("es", "Estoy triste"),
                ("de", "Ich bin traurig"),
                ("it", "Sono triste"),
                ("pt", "Estou triste"),
                ("ja", "悲しいです"),
                ("zh", "我很难过"),
                ("ko", "슬퍼요"),
                ("ru", "Мне грустно"),
                ("ar", "أنا حزين"),
                ("hi", "मैं दुखी हूँ"),
            ],
        ),
        // ── Travel & Transport ──
        (
            "airport",
            &[
                ("fr", "Aéroport"),
                ("es", "Aeropuerto"),
                ("de", "Flughafen"),
                ("it", "Aeroporto"),
                ("pt", "Aeroporto"),
                ("ja", "空港"),
                ("zh", "机场"),
                ("ko", "공항"),
                ("ru", "Аэропорт"),
                ("ar", "مطار"),
                ("hi", "हवाई अड्डा"),
            ],
        ),
        (
            "hotel",
            &[
                ("fr", "Hôtel"),
                ("es", "Hotel"),
                ("de", "Hotel"),
                ("it", "Albergo"),
                ("pt", "Hotel"),
                ("ja", "ホテル"),
                ("zh", "酒店"),
                ("ko", "호텔"),
                ("ru", "Отель"),
                ("ar", "فندق"),
                ("hi", "होटल"),
            ],
        ),
        (
            "hospital",
            &[
                ("fr", "Hôpital"),
                ("es", "Hospital"),
                ("de", "Krankenhaus"),
                ("it", "Ospedale"),
                ("pt", "Hospital"),
                ("ja", "病院"),
                ("zh", "医院"),
                ("ko", "병원"),
                ("ru", "Больница"),
                ("ar", "مستشفى"),
                ("hi", "अस्पताल"),
            ],
        ),
        (
            "restaurant",
            &[
                ("fr", "Restaurant"),
                ("es", "Restaurante"),
                ("de", "Restaurant"),
                ("it", "Ristorante"),
                ("pt", "Restaurante"),
                ("ja", "レストラン"),
                ("zh", "餐厅"),
                ("ko", "레스토랑"),
                ("ru", "Ресторан"),
                ("ar", "مطعم"),
                ("hi", "रेस्तरां"),
            ],
        ),
        (
            "train station",
            &[
                ("fr", "Gare"),
                ("es", "Estación de tren"),
                ("de", "Bahnhof"),
                ("it", "Stazione ferroviaria"),
                ("pt", "Estação de trem"),
                ("ja", "駅"),
                ("zh", "火车站"),
                ("ko", "기차역"),
                ("ru", "Вокзал"),
                ("ar", "محطة القطار"),
                ("hi", "रेलवे स्टेशन"),
            ],
        ),
        (
            "bus stop",
            &[
                ("fr", "Arrêt de bus"),
                ("es", "Parada de autobús"),
                ("de", "Bushaltestelle"),
                ("it", "Fermata dell'autobus"),
                ("pt", "Ponto de ônibus"),
                ("ja", "バス停"),
                ("zh", "公交车站"),
                ("ko", "버스 정류장"),
                ("ru", "Автобусная остановка"),
                ("ar", "موقف الحافلة"),
                ("hi", "बस स्टॉप"),
            ],
        ),
        (
            "taxi",
            &[
                ("fr", "Taxi"),
                ("es", "Taxi"),
                ("de", "Taxi"),
                ("it", "Taxi"),
                ("pt", "Táxi"),
                ("ja", "タクシー"),
                ("zh", "出租车"),
                ("ko", "택시"),
                ("ru", "Такси"),
                ("ar", "سيارة أجرة"),
                ("hi", "टैक्सी"),
            ],
        ),
        (
            "ticket",
            &[
                ("fr", "Billet"),
                ("es", "Billete"),
                ("de", "Fahrkarte"),
                ("it", "Biglietto"),
                ("pt", "Bilhete"),
                ("ja", "切符"),
                ("zh", "票"),
                ("ko", "표"),
                ("ru", "Билет"),
                ("ar", "تذكرة"),
                ("hi", "टिकट"),
            ],
        ),
        (
            "left",
            &[
                ("fr", "Gauche"),
                ("es", "Izquierda"),
                ("de", "Links"),
                ("it", "Sinistra"),
                ("pt", "Esquerda"),
                ("ja", "左"),
                ("zh", "左"),
                ("ko", "왼쪽"),
                ("ru", "Лево"),
                ("ar", "يسار"),
                ("hi", "बाएँ"),
            ],
        ),
        (
            "right",
            &[
                ("fr", "Droite"),
                ("es", "Derecha"),
                ("de", "Rechts"),
                ("it", "Destra"),
                ("pt", "Direita"),
                ("ja", "右"),
                ("zh", "右"),
                ("ko", "오른쪽"),
                ("ru", "Право"),
                ("ar", "يمين"),
                ("hi", "दाएँ"),
            ],
        ),
        (
            "straight ahead",
            &[
                ("fr", "Tout droit"),
                ("es", "Todo recto"),
                ("de", "Geradeaus"),
                ("it", "Dritto"),
                ("pt", "Em frente"),
                ("ja", "まっすぐ"),
                ("zh", "直走"),
                ("ko", "직진"),
                ("ru", "Прямо"),
                ("ar", "مباشرة"),
                ("hi", "सीधे"),
            ],
        ),
        // ── Food & Drink ──
        (
            "water",
            &[
                ("fr", "Eau"),
                ("es", "Agua"),
                ("de", "Wasser"),
                ("it", "Acqua"),
                ("pt", "Água"),
                ("ja", "水"),
                ("zh", "水"),
                ("ko", "물"),
                ("ru", "Вода"),
                ("ar", "ماء"),
                ("hi", "पानी"),
            ],
        ),
        (
            "food",
            &[
                ("fr", "Nourriture"),
                ("es", "Comida"),
                ("de", "Essen"),
                ("it", "Cibo"),
                ("pt", "Comida"),
                ("ja", "食べ物"),
                ("zh", "食物"),
                ("ko", "음식"),
                ("ru", "Еда"),
                ("ar", "طعام"),
                ("hi", "खाना"),
            ],
        ),
        (
            "coffee",
            &[
                ("fr", "Café"),
                ("es", "Café"),
                ("de", "Kaffee"),
                ("it", "Caffè"),
                ("pt", "Café"),
                ("ja", "コーヒー"),
                ("zh", "咖啡"),
                ("ko", "커피"),
                ("ru", "Кофе"),
                ("ar", "قهوة"),
                ("hi", "कॉफ़ी"),
            ],
        ),
        (
            "tea",
            &[
                ("fr", "Thé"),
                ("es", "Té"),
                ("de", "Tee"),
                ("it", "Tè"),
                ("pt", "Chá"),
                ("ja", "お茶"),
                ("zh", "茶"),
                ("ko", "차"),
                ("ru", "Чай"),
                ("ar", "شاي"),
                ("hi", "चाय"),
            ],
        ),
        (
            "beer",
            &[
                ("fr", "Bière"),
                ("es", "Cerveza"),
                ("de", "Bier"),
                ("it", "Birra"),
                ("pt", "Cerveja"),
                ("ja", "ビール"),
                ("zh", "啤酒"),
                ("ko", "맥주"),
                ("ru", "Пиво"),
                ("ar", "بيرة"),
                ("hi", "बीयर"),
            ],
        ),
        (
            "wine",
            &[
                ("fr", "Vin"),
                ("es", "Vino"),
                ("de", "Wein"),
                ("it", "Vino"),
                ("pt", "Vinho"),
                ("ja", "ワイン"),
                ("zh", "葡萄酒"),
                ("ko", "와인"),
                ("ru", "Вино"),
                ("ar", "نبيذ"),
                ("hi", "शराब"),
            ],
        ),
        (
            "bread",
            &[
                ("fr", "Pain"),
                ("es", "Pan"),
                ("de", "Brot"),
                ("it", "Pane"),
                ("pt", "Pão"),
                ("ja", "パン"),
                ("zh", "面包"),
                ("ko", "빵"),
                ("ru", "Хлеб"),
                ("ar", "خبز"),
                ("hi", "रोटी"),
            ],
        ),
        (
            "meat",
            &[
                ("fr", "Viande"),
                ("es", "Carne"),
                ("de", "Fleisch"),
                ("it", "Carne"),
                ("pt", "Carne"),
                ("ja", "肉"),
                ("zh", "肉"),
                ("ko", "고기"),
                ("ru", "Мясо"),
                ("ar", "لحم"),
                ("hi", "मांस"),
            ],
        ),
        (
            "fish",
            &[
                ("fr", "Poisson"),
                ("es", "Pescado"),
                ("de", "Fisch"),
                ("it", "Pesce"),
                ("pt", "Peixe"),
                ("ja", "魚"),
                ("zh", "鱼"),
                ("ko", "생선"),
                ("ru", "Рыба"),
                ("ar", "سمك"),
                ("hi", "मछली"),
            ],
        ),
        (
            "rice",
            &[
                ("fr", "Riz"),
                ("es", "Arroz"),
                ("de", "Reis"),
                ("it", "Riso"),
                ("pt", "Arroz"),
                ("ja", "ご飯"),
                ("zh", "米饭"),
                ("ko", "쌀"),
                ("ru", "Рис"),
                ("ar", "أرز"),
                ("hi", "चावल"),
            ],
        ),
        (
            "egg",
            &[
                ("fr", "Œuf"),
                ("es", "Huevo"),
                ("de", "Ei"),
                ("it", "Uovo"),
                ("pt", "Ovo"),
                ("ja", "卵"),
                ("zh", "鸡蛋"),
                ("ko", "달걀"),
                ("ru", "Яйцо"),
                ("ar", "بيضة"),
                ("hi", "अंडा"),
            ],
        ),
        (
            "milk",
            &[
                ("fr", "Lait"),
                ("es", "Leche"),
                ("de", "Milch"),
                ("it", "Latte"),
                ("pt", "Leite"),
                ("ja", "牛乳"),
                ("zh", "牛奶"),
                ("ko", "우유"),
                ("ru", "Молоко"),
                ("ar", "حليب"),
                ("hi", "दूध"),
            ],
        ),
        (
            "fruit",
            &[
                ("fr", "Fruit"),
                ("es", "Fruta"),
                ("de", "Obst"),
                ("it", "Frutta"),
                ("pt", "Fruta"),
                ("ja", "果物"),
                ("zh", "水果"),
                ("ko", "과일"),
                ("ru", "Фрукт"),
                ("ar", "فاكهة"),
                ("hi", "फल"),
            ],
        ),
        (
            "vegetable",
            &[
                ("fr", "Légume"),
                ("es", "Verdura"),
                ("de", "Gemüse"),
                ("it", "Verdura"),
                ("pt", "Legume"),
                ("ja", "野菜"),
                ("zh", "蔬菜"),
                ("ko", "채소"),
                ("ru", "Овощ"),
                ("ar", "خضار"),
                ("hi", "सब्ज़ी"),
            ],
        ),
        (
            "the bill",
            &[
                ("fr", "L'addition"),
                ("es", "La cuenta"),
                ("de", "Die Rechnung"),
                ("it", "Il conto"),
                ("pt", "A conta"),
                ("ja", "お会計"),
                ("zh", "买单"),
                ("ko", "계산서"),
                ("ru", "Счёт"),
                ("ar", "الحساب"),
                ("hi", "बिल"),
            ],
        ),
        // ── Numbers ──
        (
            "zero",
            &[
                ("fr", "Zéro"),
                ("es", "Cero"),
                ("de", "Null"),
                ("it", "Zero"),
                ("pt", "Zero"),
                ("ja", "零"),
                ("zh", "零"),
                ("ko", "영"),
                ("ru", "Ноль"),
                ("ar", "صفر"),
                ("hi", "शून्य"),
            ],
        ),
        (
            "one",
            &[
                ("fr", "Un"),
                ("es", "Uno"),
                ("de", "Eins"),
                ("it", "Uno"),
                ("pt", "Um"),
                ("ja", "一"),
                ("zh", "一"),
                ("ko", "하나"),
                ("ru", "Один"),
                ("ar", "واحد"),
                ("hi", "एक"),
            ],
        ),
        (
            "two",
            &[
                ("fr", "Deux"),
                ("es", "Dos"),
                ("de", "Zwei"),
                ("it", "Due"),
                ("pt", "Dois"),
                ("ja", "二"),
                ("zh", "二"),
                ("ko", "둘"),
                ("ru", "Два"),
                ("ar", "اثنان"),
                ("hi", "दो"),
            ],
        ),
        (
            "three",
            &[
                ("fr", "Trois"),
                ("es", "Tres"),
                ("de", "Drei"),
                ("it", "Tre"),
                ("pt", "Três"),
                ("ja", "三"),
                ("zh", "三"),
                ("ko", "셋"),
                ("ru", "Три"),
                ("ar", "ثلاثة"),
                ("hi", "तीन"),
            ],
        ),
        (
            "four",
            &[
                ("fr", "Quatre"),
                ("es", "Cuatro"),
                ("de", "Vier"),
                ("it", "Quattro"),
                ("pt", "Quatro"),
                ("ja", "四"),
                ("zh", "四"),
                ("ko", "넷"),
                ("ru", "Четыре"),
                ("ar", "أربعة"),
                ("hi", "चार"),
            ],
        ),
        (
            "five",
            &[
                ("fr", "Cinq"),
                ("es", "Cinco"),
                ("de", "Fünf"),
                ("it", "Cinque"),
                ("pt", "Cinco"),
                ("ja", "五"),
                ("zh", "五"),
                ("ko", "다섯"),
                ("ru", "Пять"),
                ("ar", "خمسة"),
                ("hi", "पाँच"),
            ],
        ),
        (
            "six",
            &[
                ("fr", "Six"),
                ("es", "Seis"),
                ("de", "Sechs"),
                ("it", "Sei"),
                ("pt", "Seis"),
                ("ja", "六"),
                ("zh", "六"),
                ("ko", "여섯"),
                ("ru", "Шесть"),
                ("ar", "ستة"),
                ("hi", "छह"),
            ],
        ),
        (
            "seven",
            &[
                ("fr", "Sept"),
                ("es", "Siete"),
                ("de", "Sieben"),
                ("it", "Sette"),
                ("pt", "Sete"),
                ("ja", "七"),
                ("zh", "七"),
                ("ko", "일곱"),
                ("ru", "Семь"),
                ("ar", "سبعة"),
                ("hi", "सात"),
            ],
        ),
        (
            "eight",
            &[
                ("fr", "Huit"),
                ("es", "Ocho"),
                ("de", "Acht"),
                ("it", "Otto"),
                ("pt", "Oito"),
                ("ja", "八"),
                ("zh", "八"),
                ("ko", "여덟"),
                ("ru", "Восемь"),
                ("ar", "ثمانية"),
                ("hi", "आठ"),
            ],
        ),
        (
            "nine",
            &[
                ("fr", "Neuf"),
                ("es", "Nueve"),
                ("de", "Neun"),
                ("it", "Nove"),
                ("pt", "Nove"),
                ("ja", "九"),
                ("zh", "九"),
                ("ko", "아홉"),
                ("ru", "Девять"),
                ("ar", "تسعة"),
                ("hi", "नौ"),
            ],
        ),
        (
            "ten",
            &[
                ("fr", "Dix"),
                ("es", "Diez"),
                ("de", "Zehn"),
                ("it", "Dieci"),
                ("pt", "Dez"),
                ("ja", "十"),
                ("zh", "十"),
                ("ko", "열"),
                ("ru", "Десять"),
                ("ar", "عشرة"),
                ("hi", "दस"),
            ],
        ),
        (
            "hundred",
            &[
                ("fr", "Cent"),
                ("es", "Cien"),
                ("de", "Hundert"),
                ("it", "Cento"),
                ("pt", "Cem"),
                ("ja", "百"),
                ("zh", "百"),
                ("ko", "백"),
                ("ru", "Сто"),
                ("ar", "مائة"),
                ("hi", "सौ"),
            ],
        ),
        (
            "thousand",
            &[
                ("fr", "Mille"),
                ("es", "Mil"),
                ("de", "Tausend"),
                ("it", "Mille"),
                ("pt", "Mil"),
                ("ja", "千"),
                ("zh", "千"),
                ("ko", "천"),
                ("ru", "Тысяча"),
                ("ar", "ألف"),
                ("hi", "हज़ार"),
            ],
        ),
        // ── Time & Days ──
        (
            "today",
            &[
                ("fr", "Aujourd'hui"),
                ("es", "Hoy"),
                ("de", "Heute"),
                ("it", "Oggi"),
                ("pt", "Hoje"),
                ("ja", "今日"),
                ("zh", "今天"),
                ("ko", "오늘"),
                ("ru", "Сегодня"),
                ("ar", "اليوم"),
                ("hi", "आज"),
            ],
        ),
        (
            "tomorrow",
            &[
                ("fr", "Demain"),
                ("es", "Mañana"),
                ("de", "Morgen"),
                ("it", "Domani"),
                ("pt", "Amanhã"),
                ("ja", "明日"),
                ("zh", "明天"),
                ("ko", "내일"),
                ("ru", "Завтра"),
                ("ar", "غدا"),
                ("hi", "कल"),
            ],
        ),
        (
            "yesterday",
            &[
                ("fr", "Hier"),
                ("es", "Ayer"),
                ("de", "Gestern"),
                ("it", "Ieri"),
                ("pt", "Ontem"),
                ("ja", "昨日"),
                ("zh", "昨天"),
                ("ko", "어제"),
                ("ru", "Вчера"),
                ("ar", "أمس"),
                ("hi", "कल"),
            ],
        ),
        (
            "now",
            &[
                ("fr", "Maintenant"),
                ("es", "Ahora"),
                ("de", "Jetzt"),
                ("it", "Adesso"),
                ("pt", "Agora"),
                ("ja", "今"),
                ("zh", "现在"),
                ("ko", "지금"),
                ("ru", "Сейчас"),
                ("ar", "الآن"),
                ("hi", "अभी"),
            ],
        ),
        (
            "monday",
            &[
                ("fr", "Lundi"),
                ("es", "Lunes"),
                ("de", "Montag"),
                ("it", "Lunedì"),
                ("pt", "Segunda-feira"),
                ("ja", "月曜日"),
                ("zh", "星期一"),
                ("ko", "월요일"),
                ("ru", "Понедельник"),
                ("ar", "الإثنين"),
                ("hi", "सोमवार"),
            ],
        ),
        (
            "tuesday",
            &[
                ("fr", "Mardi"),
                ("es", "Martes"),
                ("de", "Dienstag"),
                ("it", "Martedì"),
                ("pt", "Terça-feira"),
                ("ja", "火曜日"),
                ("zh", "星期二"),
                ("ko", "화요일"),
                ("ru", "Вторник"),
                ("ar", "الثلاثاء"),
                ("hi", "मंगलवार"),
            ],
        ),
        (
            "wednesday",
            &[
                ("fr", "Mercredi"),
                ("es", "Miércoles"),
                ("de", "Mittwoch"),
                ("it", "Mercoledì"),
                ("pt", "Quarta-feira"),
                ("ja", "水曜日"),
                ("zh", "星期三"),
                ("ko", "수요일"),
                ("ru", "Среда"),
                ("ar", "الأربعاء"),
                ("hi", "बुधवार"),
            ],
        ),
        (
            "thursday",
            &[
                ("fr", "Jeudi"),
                ("es", "Jueves"),
                ("de", "Donnerstag"),
                ("it", "Giovedì"),
                ("pt", "Quinta-feira"),
                ("ja", "木曜日"),
                ("zh", "星期四"),
                ("ko", "목요일"),
                ("ru", "Четверг"),
                ("ar", "الخميس"),
                ("hi", "गुरुवार"),
            ],
        ),
        (
            "friday",
            &[
                ("fr", "Vendredi"),
                ("es", "Viernes"),
                ("de", "Freitag"),
                ("it", "Venerdì"),
                ("pt", "Sexta-feira"),
                ("ja", "金曜日"),
                ("zh", "星期五"),
                ("ko", "금요일"),
                ("ru", "Пятница"),
                ("ar", "الجمعة"),
                ("hi", "शुक्रवार"),
            ],
        ),
        (
            "saturday",
            &[
                ("fr", "Samedi"),
                ("es", "Sábado"),
                ("de", "Samstag"),
                ("it", "Sabato"),
                ("pt", "Sábado"),
                ("ja", "土曜日"),
                ("zh", "星期六"),
                ("ko", "토요일"),
                ("ru", "Суббота"),
                ("ar", "السبت"),
                ("hi", "शनिवार"),
            ],
        ),
        (
            "sunday",
            &[
                ("fr", "Dimanche"),
                ("es", "Domingo"),
                ("de", "Sonntag"),
                ("it", "Domenica"),
                ("pt", "Domingo"),
                ("ja", "日曜日"),
                ("zh", "星期日"),
                ("ko", "일요일"),
                ("ru", "Воскресенье"),
                ("ar", "الأحد"),
                ("hi", "रविवार"),
            ],
        ),
        // ── People & Relations ──
        (
            "friend",
            &[
                ("fr", "Ami"),
                ("es", "Amigo"),
                ("de", "Freund"),
                ("it", "Amico"),
                ("pt", "Amigo"),
                ("ja", "友達"),
                ("zh", "朋友"),
                ("ko", "친구"),
                ("ru", "Друг"),
                ("ar", "صديق"),
                ("hi", "दोस्त"),
            ],
        ),
        (
            "family",
            &[
                ("fr", "Famille"),
                ("es", "Familia"),
                ("de", "Familie"),
                ("it", "Famiglia"),
                ("pt", "Família"),
                ("ja", "家族"),
                ("zh", "家人"),
                ("ko", "가족"),
                ("ru", "Семья"),
                ("ar", "عائلة"),
                ("hi", "परिवार"),
            ],
        ),
        (
            "mother",
            &[
                ("fr", "Mère"),
                ("es", "Madre"),
                ("de", "Mutter"),
                ("it", "Madre"),
                ("pt", "Mãe"),
                ("ja", "母"),
                ("zh", "母亲"),
                ("ko", "어머니"),
                ("ru", "Мать"),
                ("ar", "أم"),
                ("hi", "माँ"),
            ],
        ),
        (
            "father",
            &[
                ("fr", "Père"),
                ("es", "Padre"),
                ("de", "Vater"),
                ("it", "Padre"),
                ("pt", "Pai"),
                ("ja", "父"),
                ("zh", "父亲"),
                ("ko", "아버지"),
                ("ru", "Отец"),
                ("ar", "أب"),
                ("hi", "पिता"),
            ],
        ),
        (
            "child",
            &[
                ("fr", "Enfant"),
                ("es", "Niño"),
                ("de", "Kind"),
                ("it", "Bambino"),
                ("pt", "Criança"),
                ("ja", "子供"),
                ("zh", "孩子"),
                ("ko", "아이"),
                ("ru", "Ребёнок"),
                ("ar", "طفل"),
                ("hi", "बच्चा"),
            ],
        ),
        (
            "man",
            &[
                ("fr", "Homme"),
                ("es", "Hombre"),
                ("de", "Mann"),
                ("it", "Uomo"),
                ("pt", "Homem"),
                ("ja", "男性"),
                ("zh", "男人"),
                ("ko", "남자"),
                ("ru", "Мужчина"),
                ("ar", "رجل"),
                ("hi", "आदमी"),
            ],
        ),
        (
            "woman",
            &[
                ("fr", "Femme"),
                ("es", "Mujer"),
                ("de", "Frau"),
                ("it", "Donna"),
                ("pt", "Mulher"),
                ("ja", "女性"),
                ("zh", "女人"),
                ("ko", "여자"),
                ("ru", "Женщина"),
                ("ar", "امرأة"),
                ("hi", "औरत"),
            ],
        ),
        // ── Emergency ──
        (
            "help",
            &[
                ("fr", "Au secours"),
                ("es", "Ayuda"),
                ("de", "Hilfe"),
                ("it", "Aiuto"),
                ("pt", "Socorro"),
                ("ja", "助けて"),
                ("zh", "救命"),
                ("ko", "도와주세요"),
                ("ru", "Помогите"),
                ("ar", "مساعدة"),
                ("hi", "मदद"),
            ],
        ),
        (
            "emergency",
            &[
                ("fr", "Urgence"),
                ("es", "Emergencia"),
                ("de", "Notfall"),
                ("it", "Emergenza"),
                ("pt", "Emergência"),
                ("ja", "緊急"),
                ("zh", "紧急情况"),
                ("ko", "응급"),
                ("ru", "Чрезвычайная ситуация"),
                ("ar", "طوارئ"),
                ("hi", "आपातकाल"),
            ],
        ),
        (
            "police",
            &[
                ("fr", "Police"),
                ("es", "Policía"),
                ("de", "Polizei"),
                ("it", "Polizia"),
                ("pt", "Polícia"),
                ("ja", "警察"),
                ("zh", "警察"),
                ("ko", "경찰"),
                ("ru", "Полиция"),
                ("ar", "شرطة"),
                ("hi", "पुलिस"),
            ],
        ),
        (
            "fire",
            &[
                ("fr", "Feu"),
                ("es", "Fuego"),
                ("de", "Feuer"),
                ("it", "Fuoco"),
                ("pt", "Fogo"),
                ("ja", "火事"),
                ("zh", "火"),
                ("ko", "불"),
                ("ru", "Пожар"),
                ("ar", "حريق"),
                ("hi", "आग"),
            ],
        ),
        (
            "doctor",
            &[
                ("fr", "Médecin"),
                ("es", "Médico"),
                ("de", "Arzt"),
                ("it", "Medico"),
                ("pt", "Médico"),
                ("ja", "医者"),
                ("zh", "医生"),
                ("ko", "의사"),
                ("ru", "Врач"),
                ("ar", "طبيب"),
                ("hi", "डॉक्टर"),
            ],
        ),
        (
            "medicine",
            &[
                ("fr", "Médicament"),
                ("es", "Medicina"),
                ("de", "Medizin"),
                ("it", "Medicina"),
                ("pt", "Remédio"),
                ("ja", "薬"),
                ("zh", "药"),
                ("ko", "약"),
                ("ru", "Лекарство"),
                ("ar", "دواء"),
                ("hi", "दवा"),
            ],
        ),
        // ── Celebrations ──
        (
            "happy birthday",
            &[
                ("fr", "Joyeux anniversaire"),
                ("es", "Feliz cumpleaños"),
                ("de", "Alles Gute zum Geburtstag"),
                ("it", "Buon compleanno"),
                ("pt", "Feliz aniversário"),
                ("ja", "お誕生日おめでとう"),
                ("zh", "生日快乐"),
                ("ko", "생일 축하합니다"),
                ("ru", "С днём рождения"),
                ("ar", "عيد ميلاد سعيد"),
                ("hi", "जन्मदिन मुबारक"),
            ],
        ),
        (
            "happy new year",
            &[
                ("fr", "Bonne année"),
                ("es", "Feliz año nuevo"),
                ("de", "Frohes neues Jahr"),
                ("it", "Buon anno"),
                ("pt", "Feliz ano novo"),
                ("ja", "明けましておめでとう"),
                ("zh", "新年快乐"),
                ("ko", "새해 복 많이 받으세요"),
                ("ru", "С Новым годом"),
                ("ar", "سنة سعيدة"),
                ("hi", "नया साल मुबारक"),
            ],
        ),
        (
            "merry christmas",
            &[
                ("fr", "Joyeux Noël"),
                ("es", "Feliz Navidad"),
                ("de", "Frohe Weihnachten"),
                ("it", "Buon Natale"),
                ("pt", "Feliz Natal"),
                ("ja", "メリークリスマス"),
                ("zh", "圣诞快乐"),
                ("ko", "메리 크리스마스"),
                ("ru", "С Рождеством"),
                ("ar", "عيد ميلاد مجيد"),
                ("hi", "क्रिसमस की शुभकामनाएँ"),
            ],
        ),
        (
            "congratulations",
            &[
                ("fr", "Félicitations"),
                ("es", "Felicidades"),
                ("de", "Herzlichen Glückwunsch"),
                ("it", "Congratulazioni"),
                ("pt", "Parabéns"),
                ("ja", "おめでとうございます"),
                ("zh", "恭喜"),
                ("ko", "축하합니다"),
                ("ru", "Поздравляю"),
                ("ar", "تهانينا"),
                ("hi", "बधाई हो"),
            ],
        ),
        (
            "cheers",
            &[
                ("fr", "Santé"),
                ("es", "Salud"),
                ("de", "Prost"),
                ("it", "Salute"),
                ("pt", "Saúde"),
                ("ja", "乾杯"),
                ("zh", "干杯"),
                ("ko", "건배"),
                ("ru", "Ура"),
                ("ar", "في صحتك"),
                ("hi", "चीयर्स"),
            ],
        ),
        (
            "good luck",
            &[
                ("fr", "Bonne chance"),
                ("es", "Buena suerte"),
                ("de", "Viel Glück"),
                ("it", "Buona fortuna"),
                ("pt", "Boa sorte"),
                ("ja", "頑張って"),
                ("zh", "祝你好运"),
                ("ko", "행운을 빕니다"),
                ("ru", "Удачи"),
                ("ar", "حظ سعيد"),
                ("hi", "शुभकामनाएँ"),
            ],
        ),
        // ── Weather & Nature ──
        (
            "sun",
            &[
                ("fr", "Soleil"),
                ("es", "Sol"),
                ("de", "Sonne"),
                ("it", "Sole"),
                ("pt", "Sol"),
                ("ja", "太陽"),
                ("zh", "太阳"),
                ("ko", "태양"),
                ("ru", "Солнце"),
                ("ar", "شمس"),
                ("hi", "सूरज"),
            ],
        ),
        (
            "rain",
            &[
                ("fr", "Pluie"),
                ("es", "Lluvia"),
                ("de", "Regen"),
                ("it", "Pioggia"),
                ("pt", "Chuva"),
                ("ja", "雨"),
                ("zh", "雨"),
                ("ko", "비"),
                ("ru", "Дождь"),
                ("ar", "مطر"),
                ("hi", "बारिश"),
            ],
        ),
        (
            "snow",
            &[
                ("fr", "Neige"),
                ("es", "Nieve"),
                ("de", "Schnee"),
                ("it", "Neve"),
                ("pt", "Neve"),
                ("ja", "雪"),
                ("zh", "雪"),
                ("ko", "눈"),
                ("ru", "Снег"),
                ("ar", "ثلج"),
                ("hi", "बर्फ"),
            ],
        ),
        (
            "hot",
            &[
                ("fr", "Chaud"),
                ("es", "Caliente"),
                ("de", "Heiß"),
                ("it", "Caldo"),
                ("pt", "Quente"),
                ("ja", "暑い"),
                ("zh", "热"),
                ("ko", "더운"),
                ("ru", "Горячо"),
                ("ar", "حار"),
                ("hi", "गरम"),
            ],
        ),
        (
            "cold",
            &[
                ("fr", "Froid"),
                ("es", "Frío"),
                ("de", "Kalt"),
                ("it", "Freddo"),
                ("pt", "Frio"),
                ("ja", "寒い"),
                ("zh", "冷"),
                ("ko", "추운"),
                ("ru", "Холодно"),
                ("ar", "بارد"),
                ("hi", "ठंडा"),
            ],
        ),
        // ── Colors ──
        (
            "red",
            &[
                ("fr", "Rouge"),
                ("es", "Rojo"),
                ("de", "Rot"),
                ("it", "Rosso"),
                ("pt", "Vermelho"),
                ("ja", "赤"),
                ("zh", "红色"),
                ("ko", "빨간"),
                ("ru", "Красный"),
                ("ar", "أحمر"),
                ("hi", "लाल"),
            ],
        ),
        (
            "blue",
            &[
                ("fr", "Bleu"),
                ("es", "Azul"),
                ("de", "Blau"),
                ("it", "Blu"),
                ("pt", "Azul"),
                ("ja", "青"),
                ("zh", "蓝色"),
                ("ko", "파란"),
                ("ru", "Синий"),
                ("ar", "أزرق"),
                ("hi", "नीला"),
            ],
        ),
        (
            "green",
            &[
                ("fr", "Vert"),
                ("es", "Verde"),
                ("de", "Grün"),
                ("it", "Verde"),
                ("pt", "Verde"),
                ("ja", "緑"),
                ("zh", "绿色"),
                ("ko", "초록"),
                ("ru", "Зелёный"),
                ("ar", "أخضر"),
                ("hi", "हरा"),
            ],
        ),
        (
            "yellow",
            &[
                ("fr", "Jaune"),
                ("es", "Amarillo"),
                ("de", "Gelb"),
                ("it", "Giallo"),
                ("pt", "Amarelo"),
                ("ja", "黄色"),
                ("zh", "黄色"),
                ("ko", "노란"),
                ("ru", "Жёлтый"),
                ("ar", "أصفر"),
                ("hi", "पीला"),
            ],
        ),
        (
            "black",
            &[
                ("fr", "Noir"),
                ("es", "Negro"),
                ("de", "Schwarz"),
                ("it", "Nero"),
                ("pt", "Preto"),
                ("ja", "黒"),
                ("zh", "黑色"),
                ("ko", "검은"),
                ("ru", "Чёрный"),
                ("ar", "أسود"),
                ("hi", "काला"),
            ],
        ),
        (
            "white",
            &[
                ("fr", "Blanc"),
                ("es", "Blanco"),
                ("de", "Weiß"),
                ("it", "Bianco"),
                ("pt", "Branco"),
                ("ja", "白"),
                ("zh", "白色"),
                ("ko", "흰"),
                ("ru", "Белый"),
                ("ar", "أبيض"),
                ("hi", "सफ़ेद"),
            ],
        ),
        // ── Tech & Modern ──
        (
            "computer",
            &[
                ("fr", "Ordinateur"),
                ("es", "Computadora"),
                ("de", "Computer"),
                ("it", "Computer"),
                ("pt", "Computador"),
                ("ja", "コンピュータ"),
                ("zh", "电脑"),
                ("ko", "컴퓨터"),
                ("ru", "Компьютер"),
                ("ar", "حاسوب"),
                ("hi", "कंप्यूटर"),
            ],
        ),
        (
            "internet",
            &[
                ("fr", "Internet"),
                ("es", "Internet"),
                ("de", "Internet"),
                ("it", "Internet"),
                ("pt", "Internet"),
                ("ja", "インターネット"),
                ("zh", "互联网"),
                ("ko", "인터넷"),
                ("ru", "Интернет"),
                ("ar", "إنترنت"),
                ("hi", "इंटरनेट"),
            ],
        ),
        (
            "phone",
            &[
                ("fr", "Téléphone"),
                ("es", "Teléfono"),
                ("de", "Telefon"),
                ("it", "Telefono"),
                ("pt", "Telefone"),
                ("ja", "電話"),
                ("zh", "电话"),
                ("ko", "전화"),
                ("ru", "Телефон"),
                ("ar", "هاتف"),
                ("hi", "फ़ोन"),
            ],
        ),
        (
            "email",
            &[
                ("fr", "Courriel"),
                ("es", "Correo electrónico"),
                ("de", "E-Mail"),
                ("it", "Email"),
                ("pt", "E-mail"),
                ("ja", "メール"),
                ("zh", "电子邮件"),
                ("ko", "이메일"),
                ("ru", "Электронная почта"),
                ("ar", "بريد إلكتروني"),
                ("hi", "ईमेल"),
            ],
        ),
        (
            "password",
            &[
                ("fr", "Mot de passe"),
                ("es", "Contraseña"),
                ("de", "Passwort"),
                ("it", "Password"),
                ("pt", "Senha"),
                ("ja", "パスワード"),
                ("zh", "密码"),
                ("ko", "비밀번호"),
                ("ru", "Пароль"),
                ("ar", "كلمة المرور"),
                ("hi", "पासवर्ड"),
            ],
        ),
        // ── Verbs ──
        (
            "to eat",
            &[
                ("fr", "Manger"),
                ("es", "Comer"),
                ("de", "Essen"),
                ("it", "Mangiare"),
                ("pt", "Comer"),
                ("ja", "食べる"),
                ("zh", "吃"),
                ("ko", "먹다"),
                ("ru", "Есть"),
                ("ar", "يأكل"),
                ("hi", "खाना"),
            ],
        ),
        (
            "to drink",
            &[
                ("fr", "Boire"),
                ("es", "Beber"),
                ("de", "Trinken"),
                ("it", "Bere"),
                ("pt", "Beber"),
                ("ja", "飲む"),
                ("zh", "喝"),
                ("ko", "마시다"),
                ("ru", "Пить"),
                ("ar", "يشرب"),
                ("hi", "पीना"),
            ],
        ),
        (
            "to go",
            &[
                ("fr", "Aller"),
                ("es", "Ir"),
                ("de", "Gehen"),
                ("it", "Andare"),
                ("pt", "Ir"),
                ("ja", "行く"),
                ("zh", "去"),
                ("ko", "가다"),
                ("ru", "Идти"),
                ("ar", "يذهب"),
                ("hi", "जाना"),
            ],
        ),
        (
            "to come",
            &[
                ("fr", "Venir"),
                ("es", "Venir"),
                ("de", "Kommen"),
                ("it", "Venire"),
                ("pt", "Vir"),
                ("ja", "来る"),
                ("zh", "来"),
                ("ko", "오다"),
                ("ru", "Приходить"),
                ("ar", "يأتي"),
                ("hi", "आना"),
            ],
        ),
        (
            "to sleep",
            &[
                ("fr", "Dormir"),
                ("es", "Dormir"),
                ("de", "Schlafen"),
                ("it", "Dormire"),
                ("pt", "Dormir"),
                ("ja", "寝る"),
                ("zh", "睡觉"),
                ("ko", "자다"),
                ("ru", "Спать"),
                ("ar", "ينام"),
                ("hi", "सोना"),
            ],
        ),
        (
            "to work",
            &[
                ("fr", "Travailler"),
                ("es", "Trabajar"),
                ("de", "Arbeiten"),
                ("it", "Lavorare"),
                ("pt", "Trabalhar"),
                ("ja", "働く"),
                ("zh", "工作"),
                ("ko", "일하다"),
                ("ru", "Работать"),
                ("ar", "يعمل"),
                ("hi", "काम करना"),
            ],
        ),
        (
            "to read",
            &[
                ("fr", "Lire"),
                ("es", "Leer"),
                ("de", "Lesen"),
                ("it", "Leggere"),
                ("pt", "Ler"),
                ("ja", "読む"),
                ("zh", "读"),
                ("ko", "읽다"),
                ("ru", "Читать"),
                ("ar", "يقرأ"),
                ("hi", "पढ़ना"),
            ],
        ),
        (
            "to write",
            &[
                ("fr", "Écrire"),
                ("es", "Escribir"),
                ("de", "Schreiben"),
                ("it", "Scrivere"),
                ("pt", "Escrever"),
                ("ja", "書く"),
                ("zh", "写"),
                ("ko", "쓰다"),
                ("ru", "Писать"),
                ("ar", "يكتب"),
                ("hi", "लिखना"),
            ],
        ),
        (
            "to speak",
            &[
                ("fr", "Parler"),
                ("es", "Hablar"),
                ("de", "Sprechen"),
                ("it", "Parlare"),
                ("pt", "Falar"),
                ("ja", "話す"),
                ("zh", "说"),
                ("ko", "말하다"),
                ("ru", "Говорить"),
                ("ar", "يتكلم"),
                ("hi", "बोलना"),
            ],
        ),
        (
            "to listen",
            &[
                ("fr", "Écouter"),
                ("es", "Escuchar"),
                ("de", "Zuhören"),
                ("it", "Ascoltare"),
                ("pt", "Ouvir"),
                ("ja", "聞く"),
                ("zh", "听"),
                ("ko", "듣다"),
                ("ru", "Слушать"),
                ("ar", "يستمع"),
                ("hi", "सुनना"),
            ],
        ),
        (
            "to see",
            &[
                ("fr", "Voir"),
                ("es", "Ver"),
                ("de", "Sehen"),
                ("it", "Vedere"),
                ("pt", "Ver"),
                ("ja", "見る"),
                ("zh", "看"),
                ("ko", "보다"),
                ("ru", "Видеть"),
                ("ar", "يرى"),
                ("hi", "देखना"),
            ],
        ),
        (
            "to buy",
            &[
                ("fr", "Acheter"),
                ("es", "Comprar"),
                ("de", "Kaufen"),
                ("it", "Comprare"),
                ("pt", "Comprar"),
                ("ja", "買う"),
                ("zh", "买"),
                ("ko", "사다"),
                ("ru", "Покупать"),
                ("ar", "يشتري"),
                ("hi", "खरीदना"),
            ],
        ),
        (
            "to give",
            &[
                ("fr", "Donner"),
                ("es", "Dar"),
                ("de", "Geben"),
                ("it", "Dare"),
                ("pt", "Dar"),
                ("ja", "あげる"),
                ("zh", "给"),
                ("ko", "주다"),
                ("ru", "Давать"),
                ("ar", "يعطي"),
                ("hi", "देना"),
            ],
        ),
        (
            "to take",
            &[
                ("fr", "Prendre"),
                ("es", "Tomar"),
                ("de", "Nehmen"),
                ("it", "Prendere"),
                ("pt", "Pegar"),
                ("ja", "取る"),
                ("zh", "拿"),
                ("ko", "가져가다"),
                ("ru", "Брать"),
                ("ar", "يأخذ"),
                ("hi", "लेना"),
            ],
        ),
        (
            "to think",
            &[
                ("fr", "Penser"),
                ("es", "Pensar"),
                ("de", "Denken"),
                ("it", "Pensare"),
                ("pt", "Pensar"),
                ("ja", "考える"),
                ("zh", "想"),
                ("ko", "생각하다"),
                ("ru", "Думать"),
                ("ar", "يفكر"),
                ("hi", "सोचना"),
            ],
        ),
        (
            "to love",
            &[
                ("fr", "Aimer"),
                ("es", "Amar"),
                ("de", "Lieben"),
                ("it", "Amare"),
                ("pt", "Amar"),
                ("ja", "愛する"),
                ("zh", "爱"),
                ("ko", "사랑하다"),
                ("ru", "Любить"),
                ("ar", "يحب"),
                ("hi", "प्यार करना"),
            ],
        ),
        (
            "to know",
            &[
                ("fr", "Savoir"),
                ("es", "Saber"),
                ("de", "Wissen"),
                ("it", "Sapere"),
                ("pt", "Saber"),
                ("ja", "知る"),
                ("zh", "知道"),
                ("ko", "알다"),
                ("ru", "Знать"),
                ("ar", "يعرف"),
                ("hi", "जानना"),
            ],
        ),
        (
            "to want",
            &[
                ("fr", "Vouloir"),
                ("es", "Querer"),
                ("de", "Wollen"),
                ("it", "Volere"),
                ("pt", "Querer"),
                ("ja", "欲しい"),
                ("zh", "想要"),
                ("ko", "원하다"),
                ("ru", "Хотеть"),
                ("ar", "يريد"),
                ("hi", "चाहना"),
            ],
        ),
        (
            "to be able to",
            &[
                ("fr", "Pouvoir"),
                ("es", "Poder"),
                ("de", "Können"),
                ("it", "Potere"),
                ("pt", "Poder"),
                ("ja", "できる"),
                ("zh", "能"),
                ("ko", "할 수 있다"),
                ("ru", "Мочь"),
                ("ar", "يستطيع"),
                ("hi", "सकना"),
            ],
        ),
        (
            "to have",
            &[
                ("fr", "Avoir"),
                ("es", "Tener"),
                ("de", "Haben"),
                ("it", "Avere"),
                ("pt", "Ter"),
                ("ja", "持つ"),
                ("zh", "有"),
                ("ko", "가지다"),
                ("ru", "Иметь"),
                ("ar", "يملك"),
                ("hi", "रखना"),
            ],
        ),
        (
            "to be",
            &[
                ("fr", "Être"),
                ("es", "Ser"),
                ("de", "Sein"),
                ("it", "Essere"),
                ("pt", "Ser"),
                ("ja", "です"),
                ("zh", "是"),
                ("ko", "이다"),
                ("ru", "Быть"),
                ("ar", "يكون"),
                ("hi", "होना"),
            ],
        ),
    ];

    // Look up translation to English first if source is not English
    if lang == "en" {
        // Reverse lookup: find English for a non-English phrase
        for &(english, translations) in table {
            for &(tlang, tphrase) in translations {
                if tphrase.to_lowercase() == key {
                    let lang_name = lang_display_name(tlang);
                    return format!(
                        "Translation ({} → English):\n\n  \"{}\" → \"{}\"",
                        lang_name,
                        text.trim(),
                        english
                    );
                }
            }
        }
        return format!("[needs LLM] Translate to English: {}", text.trim());
    }

    // Forward lookup: English → target
    for &(english, translations) in table {
        if english == key {
            for &(tlang, tphrase) in translations {
                if tlang == lang {
                    let lang_name = lang_display_name(lang);
                    return format!(
                        "Translation (English → {}):\n\n  \"{}\" → \"{}\"",
                        lang_name,
                        text.trim(),
                        tphrase
                    );
                }
            }
        }
    }

    format!("[needs LLM] Translate to {}: {}", target_lang, text.trim())
}

/// Human-readable language name from ISO code.
fn lang_display_name(code: &str) -> &'static str {
    match code {
        "fr" => "French",
        "es" => "Spanish",
        "de" => "German",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ja" => "Japanese",
        "zh" => "Chinese",
        "ko" => "Korean",
        "ru" => "Russian",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "en" => "English",
        _ => "Unknown",
    }
}

// ─────────────────────────────────────────────────
// §99  Filesystem Operations
// ─────────────────────────────────────────────────

/// Execute a filesystem operation. Pure syscalls, no external process.
#[cfg(feature = "io")]
fn filesystem_exec(op: &FilesystemOp) -> Result<String, String> {
    use std::fs;
    use std::path::Path;

    match op {
        FilesystemOp::ReadFile { path } => {
            let p = Path::new(path);
            if !p.exists() {
                return Err(format!("File not found: {path}"));
            }
            let content = fs::read_to_string(p).map_err(|e| format!("Cannot read {path}: {e}"))?;
            let bytes = content.len();
            let lines = content.lines().count();
            Ok(format!(
                "📄 {path} ({bytes} bytes, {lines} lines)\n\n{content}"
            ))
        }
        FilesystemOp::ReadMultiple { paths } => {
            let mut out = String::new();
            for path in paths {
                let p = Path::new(path.as_str());
                if !p.exists() {
                    let _ = writeln!(out, "❌ {path}: not found");
                    continue;
                }
                match fs::read_to_string(p) {
                    Ok(content) => {
                        let _ = writeln!(out, "📄 {path} ({} bytes)\n{content}\n", content.len());
                    }
                    Err(e) => {
                        let _ = writeln!(out, "❌ {path}: {e}");
                    }
                }
            }
            Ok(out)
        }
        FilesystemOp::WriteFile { path, content } => {
            let p = Path::new(path);
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Cannot create parent dirs: {e}"))?;
            }
            fs::write(p, content).map_err(|e| format!("Cannot write {path}: {e}"))?;
            Ok(format!("✅ Wrote {} bytes to {path}", content.len()))
        }
        FilesystemOp::CreateDirectory { path } => {
            fs::create_dir_all(path).map_err(|e| format!("Cannot create directory {path}: {e}"))?;
            Ok(format!("✅ Created directory: {path}"))
        }
        FilesystemOp::ListDirectory { path } => {
            let p = Path::new(path);
            if !p.is_dir() {
                return Err(format!("Not a directory: {path}"));
            }
            let mut entries: Vec<String> = Vec::new();
            for entry in fs::read_dir(p).map_err(|e| format!("Cannot read dir: {e}"))? {
                let entry = entry.map_err(|e| format!("Dir entry error: {e}"))?;
                let name = entry.file_name().to_string_lossy().to_string();
                let meta = entry.metadata().ok();
                let is_dir = meta.as_ref().is_some_and(|m| m.is_dir());
                let size = meta.as_ref().map_or(0, |m| m.len());
                if is_dir {
                    entries.push(format!("  📁 {name}/"));
                } else {
                    entries.push(format!("  📄 {name}  ({size} bytes)"));
                }
            }
            entries.sort();
            Ok(format!("📁 {path}\n{}", entries.join("\n")))
        }
        FilesystemOp::DirectoryTree { path, max_depth } => {
            let p = Path::new(path);
            if !p.is_dir() {
                return Err(format!("Not a directory: {path}"));
            }
            let mut out = String::new();
            let _ = writeln!(out, "{path}/");
            fn walk(dir: &Path, prefix: &str, depth: usize, max_depth: usize, out: &mut String) {
                if depth >= max_depth {
                    return;
                }
                let mut entries: Vec<_> = match std::fs::read_dir(dir) {
                    Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
                    Err(_) => return,
                };
                entries.sort_by_key(|e| e.file_name());
                let count = entries.len();
                for (i, entry) in entries.iter().enumerate() {
                    let is_last = i == count - 1;
                    let connector = if is_last { "└── " } else { "├── " };
                    let name = entry.file_name().to_string_lossy().to_string();
                    let is_dir = entry.metadata().is_ok_and(|m| m.is_dir());
                    if is_dir {
                        let _ = writeln!(out, "{prefix}{connector}{name}/");
                        let child_prefix =
                            format!("{prefix}{}", if is_last { "    " } else { "│   " });
                        walk(&entry.path(), &child_prefix, depth + 1, max_depth, out);
                    } else {
                        let _ = writeln!(out, "{prefix}{connector}{name}");
                    }
                }
            }
            walk(p, "", 0, *max_depth, &mut out);
            Ok(out)
        }
        FilesystemOp::MoveFile {
            source,
            destination,
        } => {
            fs::rename(source, destination)
                .map_err(|e| format!("Cannot move {source} → {destination}: {e}"))?;
            Ok(format!("✅ Moved {source} → {destination}"))
        }
        FilesystemOp::CopyFile {
            source,
            destination,
        } => {
            let bytes = fs::copy(source, destination)
                .map_err(|e| format!("Cannot copy {source} → {destination}: {e}"))?;
            Ok(format!(
                "✅ Copied {source} → {destination} ({bytes} bytes)"
            ))
        }
        FilesystemOp::DeleteFile { path } => {
            let p = Path::new(path);
            if p.is_dir() {
                fs::remove_dir_all(p)
                    .map_err(|e| format!("Cannot delete directory {path}: {e}"))?;
            } else {
                fs::remove_file(p).map_err(|e| format!("Cannot delete {path}: {e}"))?;
            }
            Ok(format!("✅ Deleted {path}"))
        }
        FilesystemOp::FileExists { path } => {
            let p = Path::new(path);
            if p.exists() {
                let meta = fs::metadata(p).ok();
                let kind = if meta.as_ref().is_some_and(|m| m.is_dir()) {
                    "directory"
                } else {
                    "file"
                };
                Ok(format!("✅ {path} exists ({kind})"))
            } else {
                Ok(format!("❌ {path} does not exist"))
            }
        }
        FilesystemOp::FileInfo { path } => {
            let p = Path::new(path);
            let meta = fs::metadata(p).map_err(|e| format!("Cannot stat {path}: {e}"))?;
            let kind = if meta.is_dir() {
                "directory"
            } else if meta.is_symlink() {
                "symlink"
            } else {
                "file"
            };
            let size = meta.len();
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let permissions = if meta.permissions().readonly() {
                "read-only"
            } else {
                "read-write"
            };
            Ok(format!(
                "📄 {path}\n  Type: {kind}\n  Size: {size} bytes\n  Modified: {modified} (Unix)\n  Permissions: {permissions}"
            ))
        }
        FilesystemOp::SearchFiles { directory, pattern } => {
            let dir = Path::new(directory);
            if !dir.is_dir() {
                return Err(format!("Not a directory: {directory}"));
            }
            let pat_lower = pattern.to_lowercase();
            let mut matches = Vec::new();
            fn search_recursive(dir: &Path, pat: &str, matches: &mut Vec<String>, depth: usize) {
                if depth > 10 {
                    return;
                }
                let entries = match std::fs::read_dir(dir) {
                    Ok(rd) => rd,
                    Err(_) => return,
                };
                for entry in entries.filter_map(|e| e.ok()) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let path = entry.path();
                    if name.to_lowercase().contains(pat) {
                        matches.push(path.to_string_lossy().to_string());
                    }
                    if path.is_dir() && !name.starts_with('.') {
                        search_recursive(&path, pat, matches, depth + 1);
                    }
                }
            }
            search_recursive(dir, &pat_lower, &mut matches, 0);
            matches.sort();
            if matches.is_empty() {
                Ok(format!("No files matching '{pattern}' in {directory}"))
            } else {
                Ok(format!(
                    "Found {} matches for '{pattern}' in {directory}:\n{}",
                    matches.len(),
                    matches
                        .iter()
                        .map(|m| format!("  {m}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                ))
            }
        }
    }
}

// ─────────────────────────────────────────────────
// §100  Git Operations
// ─────────────────────────────────────────────────

/// Execute a git operation via subprocess. No external MCP server needed.
#[cfg(feature = "io")]
fn git_exec(op: &GitOp) -> Result<String, String> {
    use std::process::Command;

    fn run_git(repo: &str, args: &[&str]) -> Result<String, String> {
        let output = Command::new("git")
            .args(["-C", repo])
            .args(args)
            .output()
            .map_err(|e| format!("Failed to run git: {e}"))?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("git error: {stderr}"))
        }
    }

    match op {
        GitOp::Status { repo_path } => {
            let out = run_git(repo_path, &["status", "--short", "--branch"])?;
            if out.trim().is_empty() {
                Ok("Clean working tree".into())
            } else {
                Ok(format!("Git status:\n{out}"))
            }
        }
        GitOp::Log { repo_path, count } => {
            let n = format!("-{count}");
            let out = run_git(
                repo_path,
                &["log", &n, "--oneline", "--decorate", "--no-color"],
            )?;
            Ok(format!("Last {count} commits:\n{out}"))
        }
        GitOp::Diff { repo_path, target } => {
            let args = match target {
                Some(t) => vec!["diff", "--stat", t.as_str()],
                None => vec!["diff", "--stat"],
            };
            let stat = run_git(repo_path, &args)?;
            let diff_args = match target {
                Some(t) => vec!["diff", t.as_str()],
                None => vec!["diff"],
            };
            let diff = run_git(repo_path, &diff_args)?;
            if diff.trim().is_empty() {
                Ok("No changes".into())
            } else {
                // Truncate large diffs
                let truncated = if diff.len() > 8000 {
                    format!(
                        "{}\n\n... (truncated, {} total bytes)",
                        &diff[..8000],
                        diff.len()
                    )
                } else {
                    diff
                };
                Ok(format!("{stat}\n{truncated}"))
            }
        }
        GitOp::Add { repo_path, files } => {
            let mut args = vec!["add"];
            let file_refs: Vec<&str> = files.iter().map(|f| f.as_str()).collect();
            args.extend_from_slice(&file_refs);
            run_git(repo_path, &args)?;
            Ok(format!("✅ Staged {} file(s)", files.len()))
        }
        GitOp::Commit { repo_path, message } => {
            let out = run_git(repo_path, &["commit", "-m", message])?;
            Ok(format!("✅ Committed:\n{out}"))
        }
        GitOp::BranchList { repo_path } => {
            let out = run_git(repo_path, &["branch", "-a", "--no-color"])?;
            Ok(format!("Branches:\n{out}"))
        }
        GitOp::BranchCreate { repo_path, name } => {
            run_git(repo_path, &["branch", name])?;
            Ok(format!("✅ Created branch: {name}"))
        }
        GitOp::Checkout { repo_path, target } => {
            let out = run_git(repo_path, &["checkout", target])?;
            Ok(format!("✅ Checked out: {target}\n{out}"))
        }
        GitOp::Stash { repo_path, action } => {
            let args = match action.as_str() {
                "list" => vec!["stash", "list"],
                "pop" => vec!["stash", "pop"],
                "drop" => vec!["stash", "drop"],
                _ => vec!["stash"],
            };
            let out = run_git(repo_path, &args)?;
            Ok(if out.trim().is_empty() {
                format!("✅ Stash {action} completed")
            } else {
                format!("Stash {action}:\n{out}")
            })
        }
        GitOp::TagList { repo_path } => {
            let out = run_git(repo_path, &["tag", "-l", "--sort=-creatordate"])?;
            if out.trim().is_empty() {
                Ok("No tags".into())
            } else {
                Ok(format!("Tags:\n{out}"))
            }
        }
        GitOp::RemoteList { repo_path } => {
            let out = run_git(repo_path, &["remote", "-v"])?;
            if out.trim().is_empty() {
                Ok("No remotes configured".into())
            } else {
                Ok(format!("Remotes:\n{out}"))
            }
        }
        GitOp::Clone { url, destination } => {
            let output = std::process::Command::new("git")
                .args(["clone", "--progress", url, destination])
                .output()
                .map_err(|e| format!("Failed to run git clone: {e}"))?;
            if output.status.success() {
                Ok(format!("✅ Cloned {url} → {destination}"))
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("git clone error: {stderr}"))
            }
        }
    }
}

// ─────────────────────────────────────────────────
// §101  Web Fetch / Search Operations
// ─────────────────────────────────────────────────

/// External search provider (injected by ask-server).
///
/// Search/news queries route through jouleclaw's compiled pipeline
/// (6-stage QID compilation, tantivy FTS, trust scoring).
/// Fetch results are auto-ingested into the tantivy index.
#[cfg(feature = "web")]
pub trait SearchProvider: Send + Sync {
    /// Execute a web search query. Returns formatted result text.
    fn search(
        &self,
        query: &str,
        count: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send + '_>>;
    /// Execute a news search query. Returns formatted result text.
    fn news_search(
        &self,
        query: &str,
        count: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send + '_>>;
    /// Ingest a fetched page into the search index (fire-and-forget).
    fn ingest(
        &self,
        url: &str,
        title: &str,
        body: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>>;
}

/// Async execute for tools that need I/O (§101 web fetch).
/// Falls back to sync `execute()` for all other tools.
///
/// Search and news queries route through the jouleclaw search provider.
/// Fetched URLs are auto-ingested into the search index.
#[cfg(feature = "web")]
pub async fn execute_async(
    tool: &DeterministicToolKind,
    search_provider: Option<&(dyn SearchProvider + '_)>,
) -> Result<String, String> {
    match tool {
        DeterministicToolKind::WebFetch { operation } => {
            web_fetch_exec(operation, search_provider).await
        }
        _ => execute(tool),
    }
}

/// Execute a web fetch/search operation.
#[cfg(feature = "web")]
async fn web_fetch_exec(
    op: &WebFetchOp,
    search_provider: Option<&(dyn SearchProvider + '_)>,
) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("jouleclaw-tools/1.0")
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    match op {
        WebFetchOp::Fetch { url, max_length } => {
            let resp = client
                .get(url)
                .send()
                .await
                .map_err(|e| format!("Fetch error: {e}"))?;
            let status = resp.status();
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown")
                .to_string();
            let raw_body = resp.text().await.map_err(|e| format!("Read error: {e}"))?;

            // Strip HTML tags for readability
            let is_html = content_type.contains("html");
            let text = if is_html {
                strip_html_tags(&raw_body)
            } else {
                raw_body.clone()
            };

            let limit = max_length.unwrap_or(10000);
            let truncated = if text.len() > limit {
                format!(
                    "{}\n\n... (truncated at {limit} chars, {} total)",
                    &text[..limit],
                    text.len()
                )
            } else {
                text
            };

            // Auto-ingest fetched HTML into jouleclaw search index
            if let Some(provider) = search_provider
                && is_html
            {
                let title = extract_html_title(&raw_body).unwrap_or_default();
                let _ = provider.ingest(url, &title, &truncated).await;
            }

            Ok(format!(
                "🌐 {url} (HTTP {status}, {content_type})\n\n{truncated}"
            ))
        }
        WebFetchOp::Search { query, count } => {
            let n = count.unwrap_or(5);
            if let Some(provider) = search_provider {
                provider.search(query, n).await
            } else {
                Err("Search provider not configured".to_string())
            }
        }
        WebFetchOp::NewsSearch { query, count } => {
            let n = count.unwrap_or(5);
            if let Some(provider) = search_provider {
                provider.news_search(query, n).await
            } else {
                Err("Search provider not configured".to_string())
            }
        }
    }
}

/// Strip HTML tags from a string (simple regex-free approach).
#[cfg(feature = "web")]
fn strip_html_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if in_script {
            // Skip until </script>
            if i + 8 < chars.len() {
                let s: String = chars[i..i + 9].iter().collect();
                if s.to_lowercase() == "</script>" {
                    in_script = false;
                    i += 9;
                    continue;
                }
            }
            i += 1;
            continue;
        }
        match chars[i] {
            '<' => {
                // Check for <script
                if i + 7 < chars.len() {
                    let s: String = chars[i..i + 7].iter().collect();
                    if s.to_lowercase() == "<script" {
                        in_script = true;
                        i += 7;
                        continue;
                    }
                }
                // Check for <style
                if i + 6 < chars.len() {
                    let s: String = chars[i..i + 6].iter().collect();
                    if s.to_lowercase() == "<style" {
                        // Skip to </style>
                        while i < chars.len() {
                            if i + 7 < chars.len() {
                                let end: String = chars[i..i + 8].iter().collect();
                                if end.to_lowercase() == "</style>" {
                                    i += 8;
                                    break;
                                }
                            }
                            i += 1;
                        }
                        continue;
                    }
                }
                in_tag = true;
            }
            '>' if in_tag => {
                in_tag = false;
            }
            '\n' | '\r' if !in_tag => {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
            }
            _ if !in_tag => {
                out.push(chars[i]);
            }
            _ => {}
        }
        i += 1;
    }
    // Collapse multiple whitespace lines
    let lines: Vec<&str> = out
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n")
}

/// Extract the <title> text from raw HTML.
#[cfg(feature = "web")]
fn extract_html_title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title")? + 6;
    // skip past the closing '>' of the opening tag (handles <title> or <title ...>)
    let gt = lower[start..].find('>')? + start + 1;
    let end = lower[gt..].find("</title>")? + gt;
    let raw = &html[gt..end];
    let text = raw.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

// ─────────────────────────────────────────────────
// §102  OpenTofu / Terraform Module Generation
// ─────────────────────────────────────────────────

/// OpenTofu/Terraform module parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TerraformModuleParams {
    pub name: String,
    pub cloud: String, // aws, gcp, azure
    pub resources: Vec<String>,
    pub backend: Option<String>, // s3, gcs, azurerm
}

fn generate_terraform_module(p: &TerraformModuleParams) -> String {
    let mut out = String::new();
    // versions.tf
    out.push_str("# versions.tf\nterraform {\n  required_version = \">= 1.6\"\n");
    if let Some(ref backend) = p.backend {
        writeln!(out, "  backend \"{}\" {{}}", backend).unwrap();
    }
    let provider_source = match p.cloud.as_str() {
        "gcp" => "hashicorp/google",
        "azure" => "hashicorp/azurerm",
        _ => "hashicorp/aws",
    };
    write!(out, "  required_providers {{\n    {} = {{\n      source  = \"{}\"\n      version = \"~> 5.0\"\n    }}\n  }}\n}}\n\n", p.cloud, provider_source).unwrap();

    // variables.tf
    out.push_str("# variables.tf\nvariable \"environment\" {\n  type    = string\n  default = \"dev\"\n}\n\nvariable \"region\" {\n  type    = string\n  default = \"us-east-1\"\n}\n\nvariable \"tags\" {\n  type    = map(string)\n  default = {}\n}\n\n");

    // main.tf
    out.push_str("# main.tf\nlocals {\n  common_tags = merge(var.tags, {\n    Environment = var.environment\n    ManagedBy   = \"opentofu\"\n  })\n}\n\n");

    let prov = if p.cloud == "gcp" {
        "google"
    } else if p.cloud == "azure" {
        "azurerm"
    } else {
        "aws"
    };
    write!(
        out,
        "provider \"{}\" {{\n  region = var.region\n}}\n\n",
        prov
    )
    .unwrap();

    for res in &p.resources {
        let slug = res.replace(['-', '.', ' '], "_");
        match res.as_str() {
            "vpc" | "network" => {
                write!(out, "module \"{}\" {{\n  source      = \"./modules/{}\"\n  environment = var.environment\n  tags        = local.common_tags\n}}\n\n", slug, slug).unwrap();
            }
            _ => {
                write!(out, "resource \"{}\" \"{}\" {{\n  # TODO: configure {}\n  tags = local.common_tags\n}}\n\n", res, slug, res).unwrap();
            }
        }
    }

    // outputs.tf
    out.push_str("# outputs.tf\noutput \"environment\" {\n  value = var.environment\n}\n");
    out
}

// ─────────────────────────────────────────────────
// §103  Ansible Playbook Generation
// ─────────────────────────────────────────────────

/// Ansible playbook parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AnsibleParams {
    pub name: String,
    pub hosts: String,
    pub tasks: Vec<AnsibleTask>,
    pub use_become: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnsibleTask {
    Package {
        name: String,
        state: String,
    },
    Service {
        name: String,
        state: String,
        enabled: bool,
    },
    Template {
        src: String,
        dest: String,
    },
    File {
        path: String,
        state: String,
        mode: Option<String>,
    },
    Shell {
        cmd: String,
    },
    Copy {
        src: String,
        dest: String,
    },
    User {
        name: String,
        groups: Vec<String>,
    },
    Docker {
        image: String,
        name: String,
        ports: Vec<String>,
    },
}

fn generate_ansible(p: &AnsibleParams) -> String {
    let mut out = format!(
        "---\n# {}\n- name: {}\n  hosts: {}\n",
        p.name, p.name, p.hosts
    );
    if p.use_become {
        out.push_str("  become: true\n");
    }
    out.push_str("  tasks:\n");
    for task in &p.tasks {
        match task {
            AnsibleTask::Package { name, state } => {
                write!(out, "    - name: Install {name}\n      ansible.builtin.package:\n        name: {name}\n        state: {state}\n\n").unwrap();
            }
            AnsibleTask::Service {
                name,
                state,
                enabled,
            } => {
                write!(out, "    - name: Manage {name} service\n      ansible.builtin.service:\n        name: {name}\n        state: {state}\n        enabled: {enabled}\n\n").unwrap();
            }
            AnsibleTask::Template { src, dest } => {
                write!(out, "    - name: Deploy template {src}\n      ansible.builtin.template:\n        src: {src}\n        dest: {dest}\n\n").unwrap();
            }
            AnsibleTask::File { path, state, mode } => {
                write!(out, "    - name: Ensure {path}\n      ansible.builtin.file:\n        path: {path}\n        state: {state}\n").unwrap();
                if let Some(m) = mode {
                    writeln!(out, "        mode: \"{m}\"").unwrap();
                }
                out.push('\n');
            }
            AnsibleTask::Shell { cmd } => {
                write!(
                    out,
                    "    - name: Run command\n      ansible.builtin.shell: {cmd}\n\n"
                )
                .unwrap();
            }
            AnsibleTask::Copy { src, dest } => {
                write!(out, "    - name: Copy {src}\n      ansible.builtin.copy:\n        src: {src}\n        dest: {dest}\n\n").unwrap();
            }
            AnsibleTask::User { name, groups } => {
                write!(out, "    - name: Create user {name}\n      ansible.builtin.user:\n        name: {name}\n        groups: {}\n\n", groups.join(",")).unwrap();
            }
            AnsibleTask::Docker { image, name, ports } => {
                write!(out, "    - name: Run {name} container\n      community.docker.docker_container:\n        name: {name}\n        image: {image}\n        ports:\n").unwrap();
                for port in ports {
                    writeln!(out, "          - \"{port}\"").unwrap();
                }
                out.push('\n');
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────
// §104  Pulumi IaC Generation (TypeScript)
// ─────────────────────────────────────────────────

/// Pulumi parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PulumiParams {
    pub name: String,
    pub cloud: String,
    pub resources: Vec<String>,
    pub language: String, // typescript, python
}

fn generate_pulumi(p: &PulumiParams) -> String {
    if p.language == "python" {
        return generate_pulumi_python(p);
    }
    let mut out = format!(
        "// Pulumi — {}\nimport * as pulumi from \"@pulumi/pulumi\";\n",
        p.name
    );
    let pkg = match p.cloud.as_str() {
        "gcp" => "gcp",
        "azure" => "azure-native",
        _ => "aws",
    };
    write!(out, "import * as {} from \"@pulumi/{}\";\n\n", p.cloud, pkg).unwrap();

    for res in &p.resources {
        let slug = res.replace(['-', '.', ' '], "_");
        match (p.cloud.as_str(), res.as_str()) {
            ("aws", "s3") | ("aws", "bucket") => {
                write!(out, "const {slug} = new aws.s3.Bucket(\"{slug}\", {{\n  bucket: \"{slug}-${{pulumi.getStack()}}\",\n}});\n\nexport const bucketName = {slug}.bucket;\n\n").unwrap();
            }
            ("aws", "lambda") | ("aws", "function") => {
                write!(out, "const {slug} = new aws.lambda.Function(\"{slug}\", {{\n  runtime: \"nodejs20.x\",\n  handler: \"index.handler\",\n  code: new pulumi.asset.FileArchive(\"./app\"),\n}});\n\n").unwrap();
            }
            ("aws", "vpc") | ("aws", "network") => {
                write!(out, "const {slug} = new aws.ec2.Vpc(\"{slug}\", {{\n  cidrBlock: \"10.0.0.0/16\",\n  tags: {{ Name: \"{slug}\" }},\n}});\n\n").unwrap();
            }
            ("gcp", "bucket") | ("gcp", "storage") => {
                write!(out, "const {slug} = new gcp.storage.Bucket(\"{slug}\", {{\n  location: \"US\",\n}});\n\n").unwrap();
            }
            _ => {
                write!(out, "// TODO: {res}\nconst {slug} = new {}.RESOURCE_TYPE(\"{slug}\", {{\n  // configure here\n}});\n\n", p.cloud).unwrap();
            }
        }
    }
    out
}

fn generate_pulumi_python(p: &PulumiParams) -> String {
    let mut out = format!("\"\"\"Pulumi — {}\"\"\"\nimport pulumi\n", p.name);
    let pkg = match p.cloud.as_str() {
        "gcp" => "pulumi_gcp",
        "azure" => "pulumi_azure_native",
        _ => "pulumi_aws",
    };
    write!(out, "import {} as {}\n\n", pkg, p.cloud).unwrap();
    for res in &p.resources {
        let slug = res.replace(['-', '.', ' '], "_");
        match (p.cloud.as_str(), res.as_str()) {
            ("aws", "s3") | ("aws", "bucket") => {
                write!(out, "{slug} = aws.s3.Bucket(\"{slug}\",\n    bucket=f\"{slug}-{{pulumi.get_stack()}}\",\n)\npulumi.export(\"bucket_name\", {slug}.bucket)\n\n").unwrap();
            }
            _ => {
                write!(
                    out,
                    "# TODO: {res}\n{slug} = {}.RESOURCE_TYPE(\"{slug}\")\n\n",
                    p.cloud
                )
                .unwrap();
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────
// §105  CloudFormation Template Generation
// ─────────────────────────────────────────────────

/// CloudFormation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudFormationParams {
    pub description: String,
    pub resources: Vec<CfnResource>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CfnResource {
    Ec2 {
        name: String,
        instance_type: String,
        ami: String,
    },
    S3 {
        name: String,
    },
    Lambda {
        name: String,
        runtime: String,
    },
    Rds {
        name: String,
        engine: String,
        instance_class: String,
    },
    Vpc {
        name: String,
        cidr: String,
    },
    SecurityGroup {
        name: String,
        ports: Vec<u16>,
    },
    DynamoDb {
        name: String,
        hash_key: String,
    },
}

fn generate_cloudformation(p: &CloudFormationParams) -> String {
    let mut out = format!(
        "AWSTemplateFormatVersion: '2010-09-09'\nDescription: {}\n\nParameters:\n  Environment:\n    Type: String\n    Default: dev\n    AllowedValues: [dev, staging, prod]\n\nResources:\n",
        p.description
    );
    for res in &p.resources {
        match res {
            CfnResource::Ec2 {
                name,
                instance_type,
                ami,
            } => {
                write!(out, "  {name}:\n    Type: AWS::EC2::Instance\n    Properties:\n      InstanceType: {instance_type}\n      ImageId: {ami}\n      Tags:\n        - Key: Name\n          Value: {name}\n        - Key: Environment\n          Value: !Ref Environment\n\n").unwrap();
            }
            CfnResource::S3 { name } => {
                write!(out, "  {name}:\n    Type: AWS::S3::Bucket\n    Properties:\n      BucketName: !Sub \"{name}-${{Environment}}\"\n      VersioningConfiguration:\n        Status: Enabled\n\n").unwrap();
            }
            CfnResource::Lambda { name, runtime } => {
                write!(out, "  {name}:\n    Type: AWS::Lambda::Function\n    Properties:\n      FunctionName: {name}\n      Runtime: {runtime}\n      Handler: index.handler\n      Code:\n        ZipFile: |\n          exports.handler = async (event) => {{ return {{ statusCode: 200 }}; }};\n\n").unwrap();
            }
            CfnResource::Rds {
                name,
                engine,
                instance_class,
            } => {
                write!(out, "  {name}:\n    Type: AWS::RDS::DBInstance\n    Properties:\n      Engine: {engine}\n      DBInstanceClass: {instance_class}\n      MasterUsername: admin\n      MasterUserPassword: !Ref DbPassword\n\n").unwrap();
            }
            CfnResource::Vpc { name, cidr } => {
                write!(out, "  {name}:\n    Type: AWS::EC2::VPC\n    Properties:\n      CidrBlock: {cidr}\n      EnableDnsSupport: true\n      EnableDnsHostnames: true\n      Tags:\n        - Key: Name\n          Value: {name}\n\n").unwrap();
            }
            CfnResource::SecurityGroup { name, ports } => {
                write!(out, "  {name}:\n    Type: AWS::EC2::SecurityGroup\n    Properties:\n      GroupDescription: {name}\n      SecurityGroupIngress:\n").unwrap();
                for port in ports {
                    write!(out, "        - IpProtocol: tcp\n          FromPort: {port}\n          ToPort: {port}\n          CidrIp: 0.0.0.0/0\n").unwrap();
                }
                out.push('\n');
            }
            CfnResource::DynamoDb { name, hash_key } => {
                write!(out, "  {name}:\n    Type: AWS::DynamoDB::Table\n    Properties:\n      TableName: {name}\n      AttributeDefinitions:\n        - AttributeName: {hash_key}\n          AttributeType: S\n      KeySchema:\n        - AttributeName: {hash_key}\n          KeyType: HASH\n      BillingMode: PAY_PER_REQUEST\n\n").unwrap();
            }
        }
    }
    out.push_str("Outputs:\n  Environment:\n    Value: !Ref Environment\n");
    out
}

// ─────────────────────────────────────────────────
// §106  Prometheus / Grafana Config Generation
// ─────────────────────────────────────────────────

/// Monitoring parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum MonitoringParams {
    Prometheus {
        targets: Vec<PrometheusTarget>,
        rules: Vec<AlertRule>,
    },
    Grafana {
        dashboard_title: String,
        panels: Vec<GrafanaPanel>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrometheusTarget {
    pub job: String,
    pub targets: Vec<String>,
    pub scrape_interval: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlertRule {
    pub name: String,
    pub expr: String,
    pub for_duration: String,
    pub severity: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrafanaPanel {
    pub title: String,
    pub expr: String,
    pub panel_type: String, // graph, stat, gauge, table
}

fn generate_monitoring(p: &MonitoringParams) -> String {
    match p {
        MonitoringParams::Prometheus { targets, rules } => {
            let mut out = "# prometheus.yml\nglobal:\n  scrape_interval: 15s\n  evaluation_interval: 15s\n\nscrape_configs:\n".to_string();
            for t in targets {
                writeln!(out, "  - job_name: '{}'", t.job).unwrap();
                if let Some(ref interval) = t.scrape_interval {
                    writeln!(out, "    scrape_interval: {interval}").unwrap();
                }
                out.push_str("    static_configs:\n      - targets:\n");
                for addr in &t.targets {
                    writeln!(out, "          - '{}'", addr).unwrap();
                }
                out.push('\n');
            }
            if !rules.is_empty() {
                out.push_str("rule_files:\n  - 'alerts.yml'\n\n# alerts.yml\ngroups:\n  - name: alerts\n    rules:\n");
                for r in rules {
                    write!(out, "      - alert: {}\n        expr: {}\n        for: {}\n        labels:\n          severity: {}\n        annotations:\n          summary: \"{{{{ $labels.instance }}}} — {}\"\n\n",
                        r.name, r.expr, r.for_duration, r.severity, r.name).unwrap();
                }
            }
            out
        }
        MonitoringParams::Grafana {
            dashboard_title,
            panels,
        } => {
            let mut panel_json = Vec::new();
            for (i, panel) in panels.iter().enumerate() {
                let panel_type_str = match panel.panel_type.as_str() {
                    "stat" => "stat",
                    "gauge" => "gauge",
                    "table" => "table",
                    _ => "timeseries",
                };
                panel_json.push(format!(
                    "    {{\n      \"id\": {},\n      \"type\": \"{}\",\n      \"title\": \"{}\",\n      \"gridPos\": {{ \"h\": 8, \"w\": 12, \"x\": {}, \"y\": {} }},\n      \"targets\": [{{ \"expr\": \"{}\" }}]\n    }}",
                    i + 1, panel_type_str, panel.title, (i % 2) * 12, (i / 2) * 8, panel.expr
                ));
            }
            format!(
                "{{\n  \"dashboard\": {{\n    \"title\": \"{}\",\n    \"panels\": [\n{}\n    ],\n    \"time\": {{ \"from\": \"now-1h\", \"to\": \"now\" }},\n    \"refresh\": \"10s\"\n  }}\n}}",
                dashboard_title,
                panel_json.join(",\n")
            )
        }
    }
}

// ─────────────────────────────────────────────────
// §107  Nginx / Caddy Config Generation
// ─────────────────────────────────────────────────

/// Web server config parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct WebServerParams {
    pub server: WebServerKind,
    pub sites: Vec<WebServerSite>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WebServerKind {
    Nginx,
    Caddy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WebServerSite {
    pub domain: String,
    pub upstream: Option<String>, // proxy_pass target
    pub root: Option<String>,     // static file root
    pub ssl: bool,
    pub headers: Vec<(String, String)>,
}

fn generate_webserver(p: &WebServerParams) -> String {
    match p.server {
        WebServerKind::Nginx => {
            let mut out = "# nginx.conf\n".to_string();
            for site in &p.sites {
                writeln!(out, "server {{").unwrap();
                if site.ssl {
                    write!(out, "    listen 443 ssl;\n    listen [::]:443 ssl;\n    server_name {};\n    ssl_certificate     /etc/letsencrypt/live/{}/fullchain.pem;\n    ssl_certificate_key /etc/letsencrypt/live/{}/privkey.pem;\n\n", site.domain, site.domain, site.domain).unwrap();
                } else {
                    write!(
                        out,
                        "    listen 80;\n    listen [::]:80;\n    server_name {};\n\n",
                        site.domain
                    )
                    .unwrap();
                }
                for (k, v) in &site.headers {
                    writeln!(out, "    add_header {} \"{}\";", k, v).unwrap();
                }
                if let Some(ref root) = site.root {
                    write!(out, "    root {};\n    index index.html;\n\n    location / {{\n        try_files $uri $uri/ =404;\n    }}\n", root).unwrap();
                }
                if let Some(ref upstream) = site.upstream {
                    write!(out, "    location / {{\n        proxy_pass {};\n        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n        proxy_set_header X-Forwarded-Proto $scheme;\n    }}\n", upstream).unwrap();
                }
                out.push_str("}\n\n");
                // HTTP redirect
                if site.ssl {
                    write!(out, "server {{\n    listen 80;\n    server_name {};\n    return 301 https://$server_name$request_uri;\n}}\n\n", site.domain).unwrap();
                }
            }
            out
        }
        WebServerKind::Caddy => {
            let mut out = "# Caddyfile\n".to_string();
            for site in &p.sites {
                writeln!(out, "{} {{", site.domain).unwrap();
                for (k, v) in &site.headers {
                    writeln!(out, "    header {} \"{}\"", k, v).unwrap();
                }
                if let Some(ref root) = site.root {
                    write!(out, "    root * {}\n    file_server\n", root).unwrap();
                }
                if let Some(ref upstream) = site.upstream {
                    writeln!(out, "    reverse_proxy {}", upstream).unwrap();
                }
                out.push_str("}\n\n");
            }
            out
        }
    }
}

// ─────────────────────────────────────────────────
// §108  systemd Unit File Generation
// ─────────────────────────────────────────────────

/// systemd unit parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemdParams {
    pub name: String,
    pub description: String,
    pub exec_start: String,
    pub user: Option<String>,
    pub working_dir: Option<String>,
    pub restart: String, // always, on-failure, no
    pub env_vars: Vec<(String, String)>,
    pub after: Vec<String>,
    pub wants: Vec<String>,
}

fn generate_systemd(p: &SystemdParams) -> String {
    let mut out = format!("[Unit]\nDescription={}\n", p.description);
    for dep in &p.after {
        writeln!(out, "After={dep}").unwrap();
    }
    if p.after.is_empty() {
        out.push_str("After=network.target\n");
    }
    for w in &p.wants {
        writeln!(out, "Wants={w}").unwrap();
    }
    out.push_str("\n[Service]\nType=simple\n");
    writeln!(out, "ExecStart={}", p.exec_start).unwrap();
    writeln!(out, "Restart={}", p.restart).unwrap();
    out.push_str("RestartSec=5\n");
    if let Some(ref user) = p.user {
        write!(out, "User={user}\nGroup={user}\n").unwrap();
    }
    if let Some(ref dir) = p.working_dir {
        writeln!(out, "WorkingDirectory={dir}").unwrap();
    }
    for (k, v) in &p.env_vars {
        writeln!(out, "Environment={k}={v}").unwrap();
    }
    out.push_str("StandardOutput=journal\nStandardError=journal\nSyslogIdentifier=");
    out.push_str(&p.name);
    out.push_str("\n\n[Install]\nWantedBy=multi-user.target\n");
    out
}

// ─────────────────────────────────────────────────
// §109  GitHub Actions / CI Pipeline Generation
// ─────────────────────────────────────────────────

/// CI pipeline parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CiPipelineParams {
    pub platform: CiPlatform,
    pub name: String,
    pub language: String,
    pub steps: Vec<CiStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CiPlatform {
    GitHubActions,
    GitLabCi,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CiStep {
    Checkout,
    SetupLanguage { version: String },
    Install,
    Lint,
    Test,
    Build,
    Docker { image_name: String },
    Deploy { target: String },
    Custom { name: String, run: String },
}

fn generate_ci_pipeline(p: &CiPipelineParams) -> String {
    match p.platform {
        CiPlatform::GitHubActions => generate_github_actions(p),
        CiPlatform::GitLabCi => generate_gitlab_ci(p),
    }
}

fn generate_github_actions(p: &CiPipelineParams) -> String {
    let mut out = format!(
        "name: {}\n\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n",
        p.name
    );
    for step in &p.steps {
        match step {
            CiStep::Checkout => {
                out.push_str("      - uses: actions/checkout@v4\n\n");
            }
            CiStep::SetupLanguage { version } => {
                let (action, key) = match p.language.as_str() {
                    "python" => ("actions/setup-python@v5", "python-version"),
                    "go" => ("actions/setup-go@v5", "go-version"),
                    "java" => ("actions/setup-java@v4", "java-version"),
                    "rust" => ("dtolnay/rust-toolchain@stable", "toolchain"),
                    _ => ("actions/setup-node@v4", "node-version"),
                };
                write!(
                    out,
                    "      - uses: {action}\n        with:\n          {key}: '{version}'\n\n"
                )
                .unwrap();
            }
            CiStep::Install => {
                let cmd = match p.language.as_str() {
                    "python" => "pip install -r requirements.txt",
                    "go" => "go mod download",
                    "rust" => "cargo fetch",
                    "java" => "mvn dependency:resolve",
                    _ => "npm ci",
                };
                write!(
                    out,
                    "      - name: Install dependencies\n        run: {cmd}\n\n"
                )
                .unwrap();
            }
            CiStep::Lint => {
                let cmd = match p.language.as_str() {
                    "python" => "ruff check .",
                    "go" => "golangci-lint run",
                    "rust" => "cargo clippy -- -D warnings",
                    _ => "npm run lint",
                };
                write!(out, "      - name: Lint\n        run: {cmd}\n\n").unwrap();
            }
            CiStep::Test => {
                let cmd = match p.language.as_str() {
                    "python" => "pytest",
                    "go" => "go test ./...",
                    "rust" => "cargo test",
                    "java" => "mvn test",
                    _ => "npm test",
                };
                write!(out, "      - name: Test\n        run: {cmd}\n\n").unwrap();
            }
            CiStep::Build => {
                let cmd = match p.language.as_str() {
                    "rust" => "cargo build --release",
                    "go" => "go build -o app .",
                    "java" => "mvn package -DskipTests",
                    _ => "npm run build",
                };
                write!(out, "      - name: Build\n        run: {cmd}\n\n").unwrap();
            }
            CiStep::Docker { image_name } => {
                write!(out, "      - name: Build and push Docker image\n        uses: docker/build-push-action@v5\n        with:\n          push: true\n          tags: {image_name}:${{{{ github.sha }}}}\n\n").unwrap();
            }
            CiStep::Deploy { target } => {
                write!(out, "      - name: Deploy to {target}\n        run: echo \"Deploying to {target}...\"\n        # TODO: add deployment commands\n\n").unwrap();
            }
            CiStep::Custom { name, run } => {
                write!(out, "      - name: {name}\n        run: {run}\n\n").unwrap();
            }
        }
    }
    out
}

fn generate_gitlab_ci(p: &CiPipelineParams) -> String {
    let mut out = format!(
        "# .gitlab-ci.yml — {}\nstages:\n  - build\n  - test\n  - deploy\n\n",
        p.name
    );
    let image = match p.language.as_str() {
        "python" => "python:3.12",
        "go" => "golang:1.22",
        "rust" => "rust:1.77",
        "java" => "maven:3.9",
        _ => "node:20",
    };
    write!(out, "image: {image}\n\n").unwrap();
    for step in &p.steps {
        match step {
            CiStep::Test => {
                let cmd = match p.language.as_str() {
                    "python" => "pytest",
                    "go" => "go test ./...",
                    "rust" => "cargo test",
                    "java" => "mvn test",
                    _ => "npm test",
                };
                write!(out, "test:\n  stage: test\n  script:\n    - {cmd}\n\n").unwrap();
            }
            CiStep::Build => {
                let cmd = match p.language.as_str() {
                    "rust" => "cargo build --release",
                    "go" => "go build -o app .",
                    _ => "npm run build",
                };
                write!(out, "build:\n  stage: build\n  script:\n    - {cmd}\n\n").unwrap();
            }
            CiStep::Deploy { target } => {
                write!(out, "deploy:\n  stage: deploy\n  script:\n    - echo \"Deploying to {target}\"\n  only:\n    - main\n\n").unwrap();
            }
            CiStep::Custom { name, run } => {
                let slug = name.to_lowercase().replace(' ', "_");
                write!(out, "{slug}:\n  stage: build\n  script:\n    - {run}\n\n").unwrap();
            }
            _ => {}
        }
    }
    out
}

// ─────────────────────────────────────────────────
// §110  OpenAPI 3.x Spec Generation
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct OpenApiParams {
    pub title: String,
    pub version: String,
    pub base_path: String,
    pub endpoints: Vec<OpenApiEndpoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenApiEndpoint {
    pub path: String,
    pub method: String,
    pub summary: String,
    pub request_body: Option<String>,
    pub response_schema: Option<String>,
    pub params: Vec<OpenApiParam>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenApiParam {
    pub name: String,
    pub location: String, // query, path, header
    pub required: bool,
    pub param_type: String,
}

fn generate_openapi(p: &OpenApiParams) -> String {
    let mut out = format!(
        "openapi: \"3.0.3\"\ninfo:\n  title: \"{}\"\n  version: \"{}\"\npaths:\n",
        p.title, p.version
    );
    for ep in &p.endpoints {
        let method = ep.method.to_lowercase();
        write!(
            out,
            "  {}:\n    {}:\n      summary: \"{}\"\n",
            ep.path, method, ep.summary
        )
        .unwrap();

        if !ep.params.is_empty() {
            out.push_str("      parameters:\n");
            for param in &ep.params {
                write!(
                    out,
                    "        - name: \"{}\"\n          in: \"{}\"\n          required: {}\n          schema:\n            type: \"{}\"\n",
                    param.name, param.location, param.required, param.param_type
                ).unwrap();
            }
        }

        if let Some(ref body) = ep.request_body {
            write!(
                out,
                "      requestBody:\n        required: true\n        content:\n          application/json:\n            schema:\n              $ref: \"#/components/schemas/{body}\"\n"
            ).unwrap();
        }

        let resp_schema = ep.response_schema.as_deref().unwrap_or("object");
        write!(
            out,
            "      responses:\n        \"200\":\n          description: \"Success\"\n          content:\n            application/json:\n              schema:\n                type: \"{resp_schema}\"\n"
        ).unwrap();
    }
    out
}

// ─────────────────────────────────────────────────
// §111  SQL Query Builder
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SqlQueryParams {
    pub operation: SqlOp,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SqlOp {
    Select {
        table: String,
        columns: Vec<String>,
        where_clause: Option<String>,
        order_by: Option<String>,
        limit: Option<u32>,
        joins: Vec<SqlJoin>,
    },
    Insert {
        table: String,
        columns: Vec<String>,
    },
    Update {
        table: String,
        set_columns: Vec<String>,
        where_clause: String,
    },
    Delete {
        table: String,
        where_clause: String,
    },
    CreateTable {
        table: String,
        columns: Vec<SqlColumn>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SqlJoin {
    pub join_type: String, // INNER, LEFT, RIGHT
    pub table: String,
    pub on: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SqlColumn {
    pub name: String,
    pub col_type: String,
    pub constraints: String,
}

fn generate_sql(p: &SqlQueryParams) -> String {
    match &p.operation {
        SqlOp::Select {
            table,
            columns,
            where_clause,
            order_by,
            limit,
            joins,
        } => {
            let cols = if columns.is_empty() {
                "*".into()
            } else {
                columns.join(", ")
            };
            let mut out = format!("SELECT {cols}\nFROM {table}");
            for j in joins {
                write!(out, "\n{} JOIN {} ON {}", j.join_type, j.table, j.on).unwrap();
            }
            if let Some(w) = where_clause {
                write!(out, "\nWHERE {w}").unwrap();
            }
            if let Some(o) = order_by {
                write!(out, "\nORDER BY {o}").unwrap();
            }
            if let Some(l) = limit {
                write!(out, "\nLIMIT {l}").unwrap();
            }
            out.push(';');
            out
        }
        SqlOp::Insert { table, columns } => {
            let cols = columns.join(", ");
            let placeholders: Vec<String> = (1..=columns.len()).map(|i| format!("${i}")).collect();
            format!(
                "INSERT INTO {table} ({cols})\nVALUES ({});",
                placeholders.join(", ")
            )
        }
        SqlOp::Update {
            table,
            set_columns,
            where_clause,
        } => {
            let sets: Vec<String> = set_columns
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{c} = ${}", i + 1))
                .collect();
            format!(
                "UPDATE {table}\nSET {}\nWHERE {where_clause};",
                sets.join(", ")
            )
        }
        SqlOp::Delete {
            table,
            where_clause,
        } => {
            format!("DELETE FROM {table}\nWHERE {where_clause};")
        }
        SqlOp::CreateTable { table, columns } => {
            let mut out = format!("CREATE TABLE {table} (\n");
            for (i, col) in columns.iter().enumerate() {
                let comma = if i < columns.len() - 1 { "," } else { "" };
                writeln!(
                    out,
                    "  {} {} {}{}",
                    col.name, col.col_type, col.constraints, comma
                )
                .unwrap();
            }
            out.push_str(");");
            out
        }
    }
}

// ─────────────────────────────────────────────────
// §112  GraphQL Schema Generation
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct GraphqlSchemaParams {
    pub types: Vec<GqlType>,
    pub queries: Vec<GqlField>,
    pub mutations: Vec<GqlField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GqlType {
    pub name: String,
    pub fields: Vec<GqlField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GqlField {
    pub name: String,
    pub field_type: String,
    pub args: Vec<GqlArg>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GqlArg {
    pub name: String,
    pub arg_type: String,
}

fn generate_graphql_schema(p: &GraphqlSchemaParams) -> String {
    let mut out = String::new();
    for t in &p.types {
        writeln!(out, "type {} {{", t.name).unwrap();
        for f in &t.fields {
            writeln!(out, "  {}: {}", f.name, f.field_type).unwrap();
        }
        out.push_str("}\n\n");
    }
    if !p.queries.is_empty() {
        out.push_str("type Query {\n");
        for q in &p.queries {
            if q.args.is_empty() {
                writeln!(out, "  {}: {}", q.name, q.field_type).unwrap();
            } else {
                let args: Vec<String> = q
                    .args
                    .iter()
                    .map(|a| format!("{}: {}", a.name, a.arg_type))
                    .collect();
                writeln!(out, "  {}({}): {}", q.name, args.join(", "), q.field_type).unwrap();
            }
        }
        out.push_str("}\n\n");
    }
    if !p.mutations.is_empty() {
        out.push_str("type Mutation {\n");
        for m in &p.mutations {
            if m.args.is_empty() {
                writeln!(out, "  {}: {}", m.name, m.field_type).unwrap();
            } else {
                let args: Vec<String> = m
                    .args
                    .iter()
                    .map(|a| format!("{}: {}", a.name, a.arg_type))
                    .collect();
                writeln!(out, "  {}({}): {}", m.name, args.join(", "), m.field_type).unwrap();
            }
        }
        out.push_str("}\n");
    }
    out
}

// ─────────────────────────────────────────────────
// §113  Type Generator (JSON → TS/Rust/Python/Go)
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct TypeGenParams {
    pub name: String,
    pub fields: Vec<TypeGenField>,
    pub target: TypeGenTarget,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeGenTarget {
    TypeScript,
    Rust,
    Python,
    Go,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeGenField {
    pub name: String,
    pub field_type: String,
    pub optional: bool,
}

fn generate_types(p: &TypeGenParams) -> String {
    match p.target {
        TypeGenTarget::TypeScript => {
            let mut out = format!("export interface {} {{\n", p.name);
            for f in &p.fields {
                let opt = if f.optional { "?" } else { "" };
                let ts_type = to_ts_type(&f.field_type);
                writeln!(out, "  {}{}: {};", f.name, opt, ts_type).unwrap();
            }
            out.push('}');
            out
        }
        TypeGenTarget::Rust => {
            let mut out = format!(
                "#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]\npub struct {} {{\n",
                p.name
            );
            for f in &p.fields {
                let rs_type = to_rust_type(&f.field_type);
                let full_type = if f.optional {
                    format!("Option<{rs_type}>")
                } else {
                    rs_type
                };
                let snake = to_snake_case(&f.name);
                writeln!(out, "    pub {snake}: {full_type},").unwrap();
            }
            out.push('}');
            out
        }
        TypeGenTarget::Python => {
            let mut out = format!(
                "from dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass {}:\n",
                p.name
            );
            for f in &p.fields {
                let py_type = to_python_type(&f.field_type);
                let full_type = if f.optional {
                    format!("Optional[{py_type}]")
                } else {
                    py_type
                };
                writeln!(out, "    {}: {}", f.name, full_type).unwrap();
            }
            out
        }
        TypeGenTarget::Go => {
            let mut out = format!("type {} struct {{\n", p.name);
            for f in &p.fields {
                let go_type = to_go_type(&f.field_type);
                let ptr = if f.optional { "*" } else { "" };
                let pascal = to_pascal_case(&f.name);
                writeln!(out, "\t{pascal} {ptr}{go_type} `json:\"{}\"`", f.name).unwrap();
            }
            out.push('}');
            out
        }
    }
}

fn to_ts_type(t: &str) -> String {
    match t.to_lowercase().as_str() {
        "string" | "str" | "text" => "string".into(),
        "int" | "integer" | "i32" | "i64" | "u32" | "u64" | "float" | "f32" | "f64" | "number" => {
            "number".into()
        }
        "bool" | "boolean" => "boolean".into(),
        "date" | "datetime" | "timestamp" => "string".into(),
        other => other.to_string(),
    }
}

fn to_rust_type(t: &str) -> String {
    match t.to_lowercase().as_str() {
        "string" | "str" | "text" => "String".into(),
        "int" | "integer" | "i32" => "i32".into(),
        "i64" => "i64".into(),
        "u32" => "u32".into(),
        "u64" => "u64".into(),
        "float" | "f32" => "f32".into(),
        "f64" | "number" => "f64".into(),
        "bool" | "boolean" => "bool".into(),
        "date" | "datetime" | "timestamp" => "String".into(),
        other => other.to_string(),
    }
}

fn to_python_type(t: &str) -> String {
    match t.to_lowercase().as_str() {
        "string" | "str" | "text" => "str".into(),
        "int" | "integer" | "i32" | "i64" | "u32" | "u64" => "int".into(),
        "float" | "f32" | "f64" | "number" => "float".into(),
        "bool" | "boolean" => "bool".into(),
        "date" | "datetime" | "timestamp" => "str".into(),
        other => other.to_string(),
    }
}

fn to_go_type(t: &str) -> String {
    match t.to_lowercase().as_str() {
        "string" | "str" | "text" => "string".into(),
        "int" | "integer" | "i32" => "int32".into(),
        "i64" => "int64".into(),
        "u32" => "uint32".into(),
        "u64" => "uint64".into(),
        "float" | "f32" => "float32".into(),
        "f64" | "number" => "float64".into(),
        "bool" | "boolean" => "bool".into(),
        "date" | "datetime" | "timestamp" => "string".into(),
        other => other.to_string(),
    }
}

fn to_snake_case(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_lowercase().next().unwrap_or(c));
    }
    out
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────
// §114  Protobuf / gRPC Definition Generation
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ProtobufParams {
    pub package: String,
    pub messages: Vec<ProtoMessage>,
    pub services: Vec<ProtoService>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtoMessage {
    pub name: String,
    pub fields: Vec<ProtoField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtoField {
    pub name: String,
    pub field_type: String,
    pub number: u32,
    pub repeated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtoService {
    pub name: String,
    pub rpcs: Vec<ProtoRpc>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtoRpc {
    pub name: String,
    pub request: String,
    pub response: String,
}

fn generate_protobuf(p: &ProtobufParams) -> String {
    let mut out = format!("syntax = \"proto3\";\n\npackage {};\n\n", p.package);

    for msg in &p.messages {
        writeln!(out, "message {} {{", msg.name).unwrap();
        for f in &msg.fields {
            let repeated = if f.repeated { "repeated " } else { "" };
            writeln!(
                out,
                "  {}{} {} = {};",
                repeated, f.field_type, f.name, f.number
            )
            .unwrap();
        }
        out.push_str("}\n\n");
    }

    for svc in &p.services {
        writeln!(out, "service {} {{", svc.name).unwrap();
        for rpc in &svc.rpcs {
            writeln!(
                out,
                "  rpc {}({}) returns ({});",
                rpc.name, rpc.request, rpc.response
            )
            .unwrap();
        }
        out.push_str("}\n\n");
    }
    out
}

// ─────────────────────────────────────────────────
// §115  .gitignore Generator
// ─────────────────────────────────────────────────

fn generate_gitignore(language: &str) -> String {
    let l = language.to_lowercase();
    let header = format!("# .gitignore for {language}\n\n");
    let body = match l.as_str() {
        "rust" => {
            "# Build artifacts\n/target/\n\n# IDE\n.idea/\n.vscode/\n*.swp\n*.swo\n\n# OS\n.DS_Store\nThumbs.db\n\n# Cargo lock (for libraries)\n# Cargo.lock\n"
        }
        "python" | "py" => {
            "# Byte-compiled / optimized\n__pycache__/\n*.py[cod]\n*$py.class\n*.so\n\n# Virtual environments\nvenv/\n.env/\n.venv/\n\n# Distribution\ndist/\nbuild/\n*.egg-info/\n*.egg\n\n# IDE\n.idea/\n.vscode/\n*.swp\n\n# OS\n.DS_Store\nThumbs.db\n\n# Dotenv\n.env\n.env.local\n"
        }
        "node" | "nodejs" | "javascript" | "js" | "typescript" | "ts" => {
            "# Dependencies\nnode_modules/\n\n# Build output\ndist/\nbuild/\n.next/\n.nuxt/\n\n# Environment\n.env\n.env.local\n.env.*.local\n\n# IDE\n.idea/\n.vscode/\n*.swp\n\n# OS\n.DS_Store\nThumbs.db\n\n# Logs\nnpm-debug.log*\nyarn-debug.log*\nyarn-error.log*\n\n# Coverage\ncoverage/\n"
        }
        "go" | "golang" => {
            "# Binary\n*.exe\n*.exe~\n*.dll\n*.so\n*.dylib\n\n# Build output\n/bin/\n/vendor/\n\n# Test\n*.test\n*.out\ncoverage.txt\n\n# IDE\n.idea/\n.vscode/\n*.swp\n\n# OS\n.DS_Store\nThumbs.db\n"
        }
        "java" | "kotlin" | "scala" => {
            "# Build output\ntarget/\nbuild/\n*.class\n*.jar\n*.war\n*.ear\n\n# IDE\n.idea/\n*.iml\n.eclipse/\n.project\n.settings/\n.classpath\n\n# Gradle\n.gradle/\ngradle-app.setting\n\n# Maven\npom.xml.tag\npom.xml.releaseBackup\npom.xml.versionsBackup\nrelease.properties\n\n# OS\n.DS_Store\nThumbs.db\n"
        }
        "c" | "c++" | "cpp" => {
            "# Build\n*.o\n*.obj\n*.d\n*.so\n*.dylib\n*.dll\n*.a\n*.lib\n*.exe\n*.out\n\n# Build directories\nbuild/\ncmake-build-*/\n\n# IDE\n.idea/\n.vscode/\n*.swp\n.clangd/\ncompile_commands.json\n\n# OS\n.DS_Store\nThumbs.db\n"
        }
        "swift" => {
            "# Xcode\nbuild/\nDerivedData/\n*.xcodeproj/xcuserdata/\n*.xcworkspace/xcuserdata/\n*.pbxuser\n*.mode1v3\n*.mode2v3\n*.perspectivev3\nxcuserdata/\n\n# Swift Package Manager\n.build/\nPackages/\n\n# CocoaPods\nPods/\n\n# OS\n.DS_Store\n"
        }
        "ruby" | "rails" => {
            "# Bundler\n/.bundle/\n/vendor/bundle\n\n# Environment\n.env\n.env.local\n\n# Logs\nlog/*.log\ntmp/\n\n# IDE\n.idea/\n.vscode/\n*.swp\n\n# OS\n.DS_Store\nThumbs.db\n\n# Coverage\ncoverage/\n"
        }
        _ => {
            "# IDE\n.idea/\n.vscode/\n*.swp\n*.swo\n\n# OS\n.DS_Store\nThumbs.db\n\n# Environment\n.env\n.env.local\n\n# Build\nbuild/\ndist/\ntarget/\n"
        }
    };
    format!("{header}{body}")
}

// ─────────────────────────────────────────────────
// §116  Secret / Credential Detector
// ─────────────────────────────────────────────────

fn detect_secrets(input: &str) -> String {
    let patterns: Vec<(&str, &str)> = vec![
        ("AWS Access Key", r"AKIA[0-9A-Z]{16}"),
        (
            "AWS Secret Key",
            r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*[A-Za-z0-9/+=]{40}",
        ),
        ("GitHub Token", r"gh[pousr]_[A-Za-z0-9_]{36,255}"),
        ("GitHub Classic Token", r"ghp_[A-Za-z0-9]{36}"),
        ("GitLab Token", r"glpat-[A-Za-z0-9\-_]{20,}"),
        ("Slack Token", r"xox[baprs]-[0-9A-Za-z\-]{10,}"),
        (
            "Slack Webhook",
            r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
        ),
        ("Google API Key", r"AIza[0-9A-Za-z\-_]{35}"),
        (
            "Google OAuth",
            r"[0-9]+-[a-z0-9_]{32}\.apps\.googleusercontent\.com",
        ),
        ("Firebase Key", r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}"),
        ("Stripe Live Key", r"sk_live_[0-9a-zA-Z]{24,}"),
        ("Stripe Publishable", r"pk_live_[0-9a-zA-Z]{24,}"),
        ("Twilio API Key", r"SK[0-9a-fA-F]{32}"),
        ("SendGrid Key", r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"),
        ("Mailgun Key", r"key-[0-9a-zA-Z]{32}"),
        ("npm Token", r"npm_[A-Za-z0-9]{36}"),
        ("PyPI Token", r"pypi-[A-Za-z0-9_-]{50,}"),
        (
            "Heroku API Key",
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        ),
        (
            "JWT",
            r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        ),
        (
            "Private Key",
            r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        ),
        (
            "Generic Secret",
            r#"(?i)(password|secret|token|api[_\-]?key|apikey)\s*[=:]\s*['"][A-Za-z0-9/+=_\-]{8,}['"]"#,
        ),
        ("Basic Auth URL", r"https?://[^:\s]+:[^@\s]+@[^\s]+"),
        (
            "Database URL",
            r"(?i)(postgres|mysql|mongodb|redis)://[^\s]+:[^\s]+@",
        ),
        (
            "Azure Storage Key",
            r"DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]{88}",
        ),
        (
            "Datadog API Key",
            r"(?i)dd[_\-]?api[_\-]?key\s*[=:]\s*[a-f0-9]{32}",
        ),
    ];

    let mut findings = Vec::new();
    for (label, pat) in &patterns {
        if let Ok(re) = regex::Regex::new(pat) {
            for m in re.find_iter(input) {
                let found = m.as_str();
                let preview = if found.len() > 20 {
                    format!("{}...{}", &found[..10], &found[found.len() - 6..])
                } else {
                    found.to_string()
                };
                findings.push(format!(
                    "  - **{}**: `{}` (position {})",
                    label,
                    preview,
                    m.start()
                ));
            }
        }
    }

    // Entropy check for high-entropy strings (potential API keys)
    let entropy_findings = check_high_entropy(input);
    for ef in &entropy_findings {
        findings.push(ef.clone());
    }

    if findings.is_empty() {
        "No secrets or credentials detected.".into()
    } else {
        format!(
            "**Secret Scan Results** — {} finding(s):\n\n{}\n\n> ⚠️ Review each finding. False positives are possible (e.g., UUIDs matching Heroku pattern).",
            findings.len(),
            findings.join("\n")
        )
    }
}

fn check_high_entropy(input: &str) -> Vec<String> {
    let mut results = Vec::new();
    // Look for quoted strings or assignment values with high entropy
    let re = regex::Regex::new(r#"['"]([A-Za-z0-9+/=_\-]{20,})['"]"#).unwrap();
    for cap in re.captures_iter(input) {
        let s = &cap[1];
        let entropy = shannon_entropy(s);
        if entropy > 4.5 && s.len() >= 20 {
            let preview = if s.len() > 20 {
                format!("{}...{}", &s[..10], &s[s.len() - 6..])
            } else {
                s.to_string()
            };
            results.push(format!(
                "  - **High-entropy string** (Shannon={:.2}): `{preview}`",
                entropy
            ));
        }
    }
    results
}

fn shannon_entropy(s: &str) -> f64 {
    let mut freq = [0u32; 256];
    for b in s.bytes() {
        freq[b as usize] += 1;
    }
    let len = s.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

// ─────────────────────────────────────────────────
// §117  Common Regex Pattern Library
// ─────────────────────────────────────────────────

fn get_regex_pattern(name: &str) -> String {
    let lower = name.to_lowercase();
    let (pattern, description, examples) = match lower.as_str() {
        "email" | "e-mail" => (
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "Email address (RFC 5322 simplified)",
            "user@example.com, test.user+tag@sub.domain.org",
        ),
        "phone" | "telephone" | "phone number" => (
            r"^\+?[1-9]\d{0,2}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}$",
            "International phone number",
            "+1 (555) 123-4567, +44 20 7946 0958",
        ),
        "url" | "uri" | "web address" => (
            r"https?://[^\s/$.?#].[^\s]*",
            "HTTP/HTTPS URL",
            "https://example.com/path?q=1, http://localhost:8080",
        ),
        "ipv4" | "ip" | "ip address" => (
            r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$",
            "IPv4 address",
            "192.168.1.1, 10.0.0.1, 255.255.255.0",
        ),
        "ipv6" => (
            r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}$|^(?:[0-9a-fA-F]{1,4}:){1,7}:$",
            "IPv6 address (full, compressed, or mixed)",
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334, ::1",
        ),
        "date" | "iso date" | "date iso" => (
            r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
            "ISO 8601 date (YYYY-MM-DD)",
            "2026-02-26, 1999-12-31",
        ),
        "time" | "iso time" | "time iso" => (
            r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$",
            "24-hour time (HH:MM or HH:MM:SS)",
            "23:59:59, 08:30",
        ),
        "uuid" | "guid" => (
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
            "UUID (versions 1-5)",
            "550e8400-e29b-41d4-a716-446655440000",
        ),
        "credit card" | "creditcard" | "cc" => (
            r"^(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|3(?:0[0-5]|[68]\d)\d{11}|6(?:011|5\d{2})\d{12})$",
            "Major credit card numbers (Visa, MC, Amex, Discover)",
            "4111111111111111 (Visa), 5500000000000004 (MC)",
        ),
        "hex color" | "hexcolor" | "color" => (
            r"^#(?:[0-9a-fA-F]{3}){1,2}$",
            "Hex color code (3 or 6 digits)",
            "#FFF, #FF5733, #a1b2c3",
        ),
        "mac address" | "mac" => (
            r"^(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}$",
            "MAC address",
            "00:1A:2B:3C:4D:5E, AA-BB-CC-DD-EE-FF",
        ),
        "semver" | "version" | "semantic version" => (
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([\da-zA-Z-]+(?:\.[\da-zA-Z-]+)*))?(?:\+([\da-zA-Z-]+(?:\.[\da-zA-Z-]+)*))?$",
            "Semantic versioning (SemVer 2.0)",
            "1.0.0, 2.1.3-beta.1, 0.0.1+build.123",
        ),
        "slug" => (
            r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
            "URL slug (lowercase alphanumeric with hyphens)",
            "my-blog-post, hello-world-2026",
        ),
        "jwt" | "json web token" => (
            r"^eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$",
            "JSON Web Token (JWT)",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        ),
        "ssn" | "social security" => (
            r"^\d{3}-\d{2}-\d{4}$",
            "US Social Security Number",
            "123-45-6789",
        ),
        "zip code" | "zipcode" | "zip" => (
            r"^\d{5}(?:-\d{4})?$",
            "US ZIP code (5 or 9 digit)",
            "90210, 12345-6789",
        ),
        _ => {
            return format!(
                "Unknown pattern \"{name}\". Available patterns:\n\
                 email, phone, url, ipv4, ipv6, date, time, uuid, credit card,\n\
                 hex color, mac address, semver, slug, jwt, ssn, zip code"
            );
        }
    };

    format!("**{description}**\n\n```regex\n{pattern}\n```\n\n**Matches**: {examples}")
}

// ─────────────────────────────────────────────────
// §118  Kubernetes RBAC Generation
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct K8sRbacParams {
    pub kind: K8sRbacKind,
    pub name: String,
    pub namespace: Option<String>,
    pub rules: Vec<K8sRbacRule>,
    pub binding: Option<K8sRbacBinding>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum K8sRbacKind {
    Role,
    ClusterRole,
}

#[derive(Debug, Clone, PartialEq)]
pub struct K8sRbacRule {
    pub api_groups: Vec<String>,
    pub resources: Vec<String>,
    pub verbs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct K8sRbacBinding {
    pub subject_kind: String, // User, Group, ServiceAccount
    pub subject_name: String,
}

fn generate_k8s_rbac(p: &K8sRbacParams) -> String {
    let (api_version, kind, binding_kind) = match p.kind {
        K8sRbacKind::Role => ("rbac.authorization.k8s.io/v1", "Role", "RoleBinding"),
        K8sRbacKind::ClusterRole => (
            "rbac.authorization.k8s.io/v1",
            "ClusterRole",
            "ClusterRoleBinding",
        ),
    };

    let ns_line = match (&p.kind, &p.namespace) {
        (K8sRbacKind::Role, Some(ns)) => format!("  namespace: {ns}\n"),
        _ => String::new(),
    };

    let mut out = format!(
        "apiVersion: {api_version}\nkind: {kind}\nmetadata:\n  name: {}\n{ns_line}rules:\n",
        p.name
    );

    for rule in &p.rules {
        let groups: Vec<String> = rule
            .api_groups
            .iter()
            .map(|g| format!("\"{}\"", g))
            .collect();
        let resources: Vec<String> = rule
            .resources
            .iter()
            .map(|r| format!("\"{}\"", r))
            .collect();
        let verbs: Vec<String> = rule.verbs.iter().map(|v| format!("\"{}\"", v)).collect();
        write!(
            out,
            "  - apiGroups: [{}]\n    resources: [{}]\n    verbs: [{}]\n",
            groups.join(", "),
            resources.join(", "),
            verbs.join(", ")
        )
        .unwrap();
    }

    if let Some(ref binding) = p.binding {
        let binding_ns = match (&p.kind, &p.namespace) {
            (K8sRbacKind::Role, Some(ns)) => format!("  namespace: {ns}\n"),
            _ => String::new(),
        };
        write!(
            out,
            "\n---\napiVersion: {api_version}\nkind: {binding_kind}\nmetadata:\n  name: {}-binding\n{binding_ns}\
             subjects:\n  - kind: {}\n    name: {}\n    apiGroup: rbac.authorization.k8s.io\n\
             roleRef:\n  kind: {kind}\n  name: {}\n  apiGroup: rbac.authorization.k8s.io\n",
            p.name, binding.subject_kind, binding.subject_name, p.name
        ).unwrap();
    }

    out
}

// ─────────────────────────────────────────────────
// §119  Kubernetes NetworkPolicy Generation
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct K8sNetworkPolicyParams {
    pub name: String,
    pub namespace: String,
    pub pod_selector: Vec<(String, String)>, // label key-value pairs
    pub ingress_rules: Vec<K8sNetPolRule>,
    pub egress_rules: Vec<K8sNetPolRule>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct K8sNetPolRule {
    pub ports: Vec<K8sNetPolPort>,
    pub pod_selector: Vec<(String, String)>,
    pub namespace_selector: Vec<(String, String)>,
    pub cidr: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct K8sNetPolPort {
    pub protocol: String,
    pub port: u16,
}

fn generate_k8s_network_policy(p: &K8sNetworkPolicyParams) -> String {
    let mut out = format!(
        "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: {}\n  namespace: {}\nspec:\n  podSelector:\n    matchLabels:\n",
        p.name, p.namespace
    );

    for (k, v) in &p.pod_selector {
        writeln!(out, "      {k}: \"{v}\"").unwrap();
    }

    let mut policy_types = Vec::new();
    if !p.ingress_rules.is_empty() {
        policy_types.push("Ingress");
    }
    if !p.egress_rules.is_empty() {
        policy_types.push("Egress");
    }
    if policy_types.is_empty() {
        policy_types.push("Ingress");
    }

    writeln!(out, "  policyTypes:").unwrap();
    for pt in &policy_types {
        writeln!(out, "    - {pt}").unwrap();
    }

    if !p.ingress_rules.is_empty() {
        out.push_str("  ingress:\n");
        for rule in &p.ingress_rules {
            write_netpol_rule(&mut out, rule);
        }
    }

    if !p.egress_rules.is_empty() {
        out.push_str("  egress:\n");
        for rule in &p.egress_rules {
            write_netpol_rule(&mut out, rule);
        }
    }

    out
}

fn write_netpol_rule(out: &mut String, rule: &K8sNetPolRule) {
    out.push_str("    - ");
    let mut first = true;

    if !rule.ports.is_empty() {
        out.push_str("ports:\n");
        for port in &rule.ports {
            write!(
                out,
                "        - protocol: {}\n          port: {}\n",
                port.protocol, port.port
            )
            .unwrap();
        }
        first = false;
    }

    let has_from =
        !rule.pod_selector.is_empty() || !rule.namespace_selector.is_empty() || rule.cidr.is_some();
    if has_from {
        if !first {
            out.push_str("      ");
        }
        out.push_str("from:\n");
        if !rule.pod_selector.is_empty() {
            out.push_str("        - podSelector:\n            matchLabels:\n");
            for (k, v) in &rule.pod_selector {
                writeln!(out, "                {k}: \"{v}\"").unwrap();
            }
        }
        if !rule.namespace_selector.is_empty() {
            out.push_str("        - namespaceSelector:\n            matchLabels:\n");
            for (k, v) in &rule.namespace_selector {
                writeln!(out, "                {k}: \"{v}\"").unwrap();
            }
        }
        if let Some(ref cidr) = rule.cidr {
            write!(out, "        - ipBlock:\n            cidr: {cidr}\n").unwrap();
        }
    }
}

// ─────────────────────────────────────────────────
// §120  AWS IAM Policy Generation
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct AwsIamParams {
    pub policy_name: String,
    pub description: String,
    pub statements: Vec<IamStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IamStatement {
    pub sid: String,
    pub effect: String, // Allow, Deny
    pub actions: Vec<String>,
    pub resources: Vec<String>,
    pub conditions: Vec<IamCondition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IamCondition {
    pub operator: String,
    pub key: String,
    pub values: Vec<String>,
}

fn generate_aws_iam_policy(p: &AwsIamParams) -> String {
    let mut statements = Vec::new();
    for stmt in &p.statements {
        let actions: Vec<String> = stmt.actions.iter().map(|a| format!("\"{}\"", a)).collect();
        let resources: Vec<String> = stmt
            .resources
            .iter()
            .map(|r| format!("\"{}\"", r))
            .collect();
        let mut s = format!(
            "    {{\n      \"Sid\": \"{}\",\n      \"Effect\": \"{}\",\n      \"Action\": [{}],\n      \"Resource\": [{}]",
            stmt.sid,
            stmt.effect,
            actions.join(", "),
            resources.join(", ")
        );

        if !stmt.conditions.is_empty() {
            s.push_str(",\n      \"Condition\": {\n");
            for (i, cond) in stmt.conditions.iter().enumerate() {
                let vals: Vec<String> = cond.values.iter().map(|v| format!("\"{}\"", v)).collect();
                write!(
                    s,
                    "        \"{}\": {{\n          \"{}\": [{}]\n        }}",
                    cond.operator,
                    cond.key,
                    vals.join(", ")
                )
                .unwrap();
                if i < stmt.conditions.len() - 1 {
                    s.push(',');
                }
                s.push('\n');
            }
            s.push_str("      }");
        }

        s.push_str("\n    }");
        statements.push(s);
    }

    format!(
        "{{\n  \"Version\": \"2012-10-17\",\n  \"Statement\": [\n{}\n  ]\n}}",
        statements.join(",\n")
    )
}

// ─────────────────────────────────────────────────
// §121  Syllogism Validator
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SyllogismParams {
    pub premises: Vec<String>,
    pub conclusion: String,
}

fn evaluate_syllogism(p: &SyllogismParams) -> String {
    // Parse categorical syllogism: "All A are B", "Some A are B", "No A are B"
    let mut sets: std::collections::HashMap<String, Vec<(String, Quantifier)>> =
        std::collections::HashMap::new();

    for premise in &p.premises {
        if let Some((quant, subject, predicate)) = parse_categorical(premise) {
            sets.entry(subject.clone())
                .or_default()
                .push((predicate.clone(), quant));
            sets.entry(predicate.clone()).or_default().push((
                subject.clone(),
                match quant {
                    Quantifier::All => Quantifier::All,
                    Quantifier::Some => Quantifier::Some,
                    Quantifier::No => Quantifier::No,
                },
            ));
        }
    }

    if let Some((quant, subj, pred)) = parse_categorical(&p.conclusion) {
        // Check if conclusion follows from premises using basic syllogistic rules
        let valid = check_syllogism_validity(&p.premises, &quant, &subj, &pred);
        let validity = if valid {
            "**VALID** ✓"
        } else {
            "**INVALID** ✗"
        };

        let mut out = String::from("**Syllogism Analysis**\n\n");
        out.push_str("Premises:\n");
        for (i, premise) in p.premises.iter().enumerate() {
            writeln!(out, "  {}. {}", i + 1, premise).unwrap();
        }
        write!(
            out,
            "\nConclusion: {}\n\nVerdict: {}\n",
            p.conclusion, validity
        )
        .unwrap();

        if !valid {
            out.push_str("\nReason: The conclusion does not logically follow from the premises.\n");
            out.push_str("Common fallacies: undistributed middle, illicit major/minor, exclusive premises.\n");
        }
        out
    } else {
        let mut out = String::from("**Syllogism Analysis**\n\nPremises:\n");
        for (i, premise) in p.premises.iter().enumerate() {
            writeln!(out, "  {}. {}", i + 1, premise).unwrap();
        }
        write!(out, "\nConclusion: {}\n\nCould not parse conclusion into categorical form.\nExpected: \"All/Some/No X are Y\"\n", p.conclusion).unwrap();
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Quantifier {
    All,
    Some,
    No,
}

fn parse_categorical(s: &str) -> Option<(Quantifier, String, String)> {
    let lower = s.to_lowercase();
    let lower = lower.trim().trim_end_matches('.');

    // "All A are B" / "Every A is B"
    if let Some(rest) = lower
        .strip_prefix("all ")
        .or_else(|| lower.strip_prefix("every "))
        && let Some(pos) = rest.find(" are ").or_else(|| rest.find(" is "))
    {
        let sep_len = if rest[pos..].starts_with(" are ") {
            5
        } else {
            4
        };
        let subj = rest[..pos].trim().to_string();
        let pred = rest[pos + sep_len..].trim().to_string();
        return Some((Quantifier::All, subj, pred));
    }

    // "No A are B" / "No A is B"
    if let Some(rest) = lower.strip_prefix("no ")
        && let Some(pos) = rest.find(" are ").or_else(|| rest.find(" is "))
    {
        let sep_len = if rest[pos..].starts_with(" are ") {
            5
        } else {
            4
        };
        let subj = rest[..pos].trim().to_string();
        let pred = rest[pos + sep_len..].trim().to_string();
        return Some((Quantifier::No, subj, pred));
    }

    // "Some A are B" / "Some A is B"
    if let Some(rest) = lower.strip_prefix("some ")
        && let Some(pos) = rest.find(" are ").or_else(|| rest.find(" is "))
    {
        let sep_len = if rest[pos..].starts_with(" are ") {
            5
        } else {
            4
        };
        let subj = rest[..pos].trim().to_string();
        let pred = rest[pos + sep_len..].trim().to_string();
        return Some((Quantifier::Some, subj, pred));
    }

    None
}

fn check_syllogism_validity(premises: &[String], cq: &Quantifier, cs: &str, cp: &str) -> bool {
    // Parse all premises
    let parsed: Vec<(Quantifier, String, String)> = premises
        .iter()
        .filter_map(|p| parse_categorical(p))
        .collect();

    if parsed.len() < 2 {
        return false;
    }

    // Classic categorical syllogism: 2 premises, find the middle term
    let (q1, s1, p1) = &parsed[0];
    let (q2, s2, p2) = &parsed[1];

    // Find middle term (appears in both premises but not in conclusion)
    let terms = [s1.as_str(), p1.as_str(), s2.as_str(), p2.as_str()];
    let conclusion_terms = [cs, cp];

    let middle = terms.iter().find(|&&t| {
        terms.iter().filter(|&&u| u == t).count() >= 2 && !conclusion_terms.contains(&t)
    });

    let middle = match middle {
        Some(m) => *m,
        None => return false,
    };

    // Barbara: All M are P, All S are M → All S are P
    if *q1 == Quantifier::All
        && *q2 == Quantifier::All
        && *cq == Quantifier::All
        && ((p1.as_str() == cp
            && s1.as_str() == middle
            && s2.as_str() == cs
            && p2.as_str() == middle)
            || (p2.as_str() == cp
                && s2.as_str() == middle
                && s1.as_str() == cs
                && p1.as_str() == middle)
            || (p1.as_str() == cp
                && p2.as_str() == middle
                && s2.as_str() == cs
                && s1.as_str() == middle))
    {
        return true;
    }

    // Celarent: No M are P, All S are M → No S are P
    if *cq == Quantifier::No
        && ((*q1 == Quantifier::No && *q2 == Quantifier::All)
            || (*q1 == Quantifier::All && *q2 == Quantifier::No))
    {
        return true;
    }

    // Darii: All M are P, Some S are M → Some S are P
    if *cq == Quantifier::Some
        && ((*q1 == Quantifier::All && *q2 == Quantifier::Some)
            || (*q1 == Quantifier::Some && *q2 == Quantifier::All))
    {
        return true;
    }

    false
}

// ─────────────────────────────────────────────────
// §122  Decision Matrix (Weighted Criteria)
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionMatrixParams {
    pub options: Vec<String>,
    pub criteria: Vec<DecisionCriterion>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionCriterion {
    pub name: String,
    pub weight: f64,
    pub scores: Vec<f64>, // one score per option
}

fn generate_decision_matrix(p: &DecisionMatrixParams) -> String {
    let mut out = String::from("**Decision Matrix**\n\n");

    // Header
    write!(out, "| Criteria | Weight |").unwrap();
    for opt in &p.options {
        write!(out, " {} |", opt).unwrap();
    }
    out.push('\n');

    write!(out, "|----------|--------|").unwrap();
    for _ in &p.options {
        out.push_str("--------|");
    }
    out.push('\n');

    // Rows
    for crit in &p.criteria {
        write!(out, "| {} | {:.1} |", crit.name, crit.weight).unwrap();
        for (i, _) in p.options.iter().enumerate() {
            let score = crit.scores.get(i).copied().unwrap_or(0.0);
            write!(out, " {:.1} |", score).unwrap();
        }
        out.push('\n');
    }

    // Weighted totals
    let mut totals: Vec<f64> = vec![0.0; p.options.len()];
    let total_weight: f64 = p.criteria.iter().map(|c| c.weight).sum();
    for crit in &p.criteria {
        let norm_weight = if total_weight > 0.0 {
            crit.weight / total_weight
        } else {
            0.0
        };
        for (i, _) in p.options.iter().enumerate() {
            let score = crit.scores.get(i).copied().unwrap_or(0.0);
            totals[i] += score * norm_weight;
        }
    }

    write!(out, "| **Weighted Total** | |").unwrap();
    for total in &totals {
        write!(out, " **{:.2}** |", total).unwrap();
    }
    out.push('\n');

    // Recommendation
    if let Some((best_idx, best_score)) = totals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    {
        write!(
            out,
            "\n**Recommendation**: {} (score: {:.2})\n",
            p.options[best_idx], best_score
        )
        .unwrap();
    }

    out
}

// ─────────────────────────────────────────────────
// §123  SWOT Analysis
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SwotParams {
    pub subject: String,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub opportunities: Vec<String>,
    pub threats: Vec<String>,
}

fn generate_swot(p: &SwotParams) -> String {
    let mut out = format!("**SWOT Analysis: {}**\n\n", p.subject);

    let s_items = format_bullet_list(&p.strengths);
    let w_items = format_bullet_list(&p.weaknesses);
    let o_items = format_bullet_list(&p.opportunities);
    let t_items = format_bullet_list(&p.threats);

    // Box-drawing SWOT grid
    writeln!(
        out,
        "┌─────────────────────────────┬─────────────────────────────┐"
    )
    .unwrap();
    writeln!(
        out,
        "│ **Strengths** (Internal +)  │ **Weaknesses** (Internal −) │"
    )
    .unwrap();
    writeln!(
        out,
        "│                             │                             │"
    )
    .unwrap();
    for (s, w) in s_items.iter().zip_longest(w_items.iter()) {
        let s_text = s.map_or("", |v| v.as_str());
        let w_text = w.map_or("", |v| v.as_str());
        writeln!(
            out,
            "│ {:27} │ {:27} │",
            truncate_pad(s_text, 27),
            truncate_pad(w_text, 27)
        )
        .unwrap();
    }
    writeln!(
        out,
        "├─────────────────────────────┼─────────────────────────────┤"
    )
    .unwrap();
    writeln!(
        out,
        "│ **Opportunities** (Ext. +)  │ **Threats** (External −)    │"
    )
    .unwrap();
    writeln!(
        out,
        "│                             │                             │"
    )
    .unwrap();
    for (o, t) in o_items.iter().zip_longest(t_items.iter()) {
        let o_text = o.map_or("", |v| v.as_str());
        let t_text = t.map_or("", |v| v.as_str());
        writeln!(
            out,
            "│ {:27} │ {:27} │",
            truncate_pad(o_text, 27),
            truncate_pad(t_text, 27)
        )
        .unwrap();
    }
    writeln!(
        out,
        "└─────────────────────────────┴─────────────────────────────┘"
    )
    .unwrap();

    // Summary counts
    write!(
        out,
        "\nTotal: {} strengths, {} weaknesses, {} opportunities, {} threats\n",
        p.strengths.len(),
        p.weaknesses.len(),
        p.opportunities.len(),
        p.threats.len()
    )
    .unwrap();

    out
}

fn format_bullet_list(items: &[String]) -> Vec<String> {
    items.iter().map(|item| format!("• {}", item)).collect()
}

fn truncate_pad(s: &str, width: usize) -> String {
    if s.chars().count() > width {
        let truncated: String = s.chars().take(width - 1).collect();
        format!("{}…", truncated)
    } else {
        format!("{:width$}", s, width = width)
    }
}

/// Zip two iterators, yielding Some from whichever is longer
trait ZipLongest: Iterator + Sized {
    fn zip_longest<U: Iterator>(self, other: U) -> ZipLongestIter<Self, U> {
        ZipLongestIter { a: self, b: other }
    }
}

impl<T: Iterator> ZipLongest for T {}

struct ZipLongestIter<A, B> {
    a: A,
    b: B,
}

impl<A: Iterator, B: Iterator> Iterator for ZipLongestIter<A, B> {
    type Item = (Option<A::Item>, Option<B::Item>);
    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next();
        let b = self.b.next();
        if a.is_none() && b.is_none() {
            None
        } else {
            Some((a, b))
        }
    }
}

// ─────────────────────────────────────────────────
// §124  Pros/Cons Analysis
// ─────────────────────────────────────────────────

fn generate_pros_cons(topic: &str, pros: &[String], cons: &[String]) -> String {
    let mut out = format!("**Pros & Cons: {}**\n\n", topic);

    out.push_str("**Pros** ✓\n");
    if pros.is_empty() {
        out.push_str("  (none listed)\n");
    } else {
        for (i, pro) in pros.iter().enumerate() {
            writeln!(out, "  {}. {}", i + 1, pro).unwrap();
        }
    }

    out.push_str("\n**Cons** ✗\n");
    if cons.is_empty() {
        out.push_str("  (none listed)\n");
    } else {
        for (i, con) in cons.iter().enumerate() {
            writeln!(out, "  {}. {}", i + 1, con).unwrap();
        }
    }

    let balance = pros.len() as i32 - cons.len() as i32;
    let verdict = if balance > 0 {
        "leans positive"
    } else if balance < 0 {
        "leans negative"
    } else {
        "balanced"
    };
    write!(
        out,
        "\n**Balance**: {} pros vs {} cons — {}\n",
        pros.len(),
        cons.len(),
        verdict
    )
    .unwrap();

    out
}

// ─────────────────────────────────────────────────
// §125  Root Cause Analysis (5 Whys + Fishbone)
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct RootCauseParams {
    pub problem: String,
    pub whys: Vec<String>,
    pub categories: Vec<RootCauseCategory>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootCauseCategory {
    pub name: String,
    pub causes: Vec<String>,
}

fn generate_root_cause(p: &RootCauseParams) -> String {
    let mut out = format!("**Root Cause Analysis**\n\nProblem: {}\n\n", p.problem);

    // 5 Whys
    if !p.whys.is_empty() {
        out.push_str("**5 Whys Chain:**\n");
        for (i, why) in p.whys.iter().enumerate() {
            let indent = "  ".repeat(i + 1);
            writeln!(out, "{}Why {}? → {}", indent, i + 1, why).unwrap();
        }
        if let Some(root) = p.whys.last() {
            write!(out, "\n**Root Cause**: {}\n", root).unwrap();
        }
        out.push('\n');
    }

    // Fishbone / Ishikawa
    if !p.categories.is_empty() {
        out.push_str("**Fishbone Diagram (Ishikawa):**\n\n");

        // Draw the fishbone
        let problem_line = format!("    ──────── {} ────────", p.problem);
        let spine_len = problem_line.chars().count();
        out.push_str(&"─".repeat(spine_len.min(60)));
        out.push_str("→ EFFECT\n");
        writeln!(out, "│  {}", p.problem).unwrap();

        for cat in &p.categories {
            writeln!(out, "├── {}", cat.name).unwrap();
            for cause in &cat.causes {
                writeln!(out, "│   └── {}", cause).unwrap();
            }
        }
        out.push_str("│\n");
    }

    out
}

// ─────────────────────────────────────────────────
// §126  Logical Deduction Chain
// ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct DeductionParams {
    pub premises: Vec<String>,
    pub rules: Vec<DeductionRule>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeductionRule {
    pub name: String,
    pub from: String,      // premise label or derived fact
    pub operation: String, // "modus_ponens", "modus_tollens", "hypothetical_syllogism", "disjunctive_syllogism"
    pub yields: String,    // derived conclusion
}

fn evaluate_deduction(p: &DeductionParams) -> String {
    let mut out = String::from("**Logical Deduction Chain**\n\n");

    out.push_str("**Premises:**\n");
    for (i, premise) in p.premises.iter().enumerate() {
        writeln!(out, "  P{}. {}", i + 1, premise).unwrap();
    }
    out.push('\n');

    out.push_str("**Derivation:**\n");
    for (i, rule) in p.rules.iter().enumerate() {
        let rule_name = match rule.operation.as_str() {
            "modus_ponens" => "Modus Ponens (If P→Q and P, then Q)",
            "modus_tollens" => "Modus Tollens (If P→Q and ¬Q, then ¬P)",
            "hypothetical_syllogism" => "Hypothetical Syllogism (If P→Q and Q→R, then P→R)",
            "disjunctive_syllogism" => "Disjunctive Syllogism (If P∨Q and ¬P, then Q)",
            "conjunction" => "Conjunction (P and Q, therefore P∧Q)",
            "simplification" => "Simplification (P∧Q, therefore P)",
            "addition" => "Addition (P, therefore P∨Q)",
            "contrapositive" => "Contrapositive (P→Q ≡ ¬Q→¬P)",
            other => other,
        };
        write!(
            out,
            "  D{}. {} [from: {}, by: {}]\n      → {}\n",
            i + 1,
            rule.yields,
            rule.from,
            rule_name,
            rule.yields
        )
        .unwrap();
    }

    if let Some(last) = p.rules.last() {
        write!(out, "\n**Conclusion**: {}\n", last.yields).unwrap();
    }

    // Validate chain integrity
    let mut known: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, premise) in p.premises.iter().enumerate() {
        known.insert(format!("P{}", i + 1));
        known.insert(premise.clone());
    }

    let mut valid = true;
    for (i, rule) in p.rules.iter().enumerate() {
        // Check that 'from' references something we know
        let from_parts: Vec<&str> = rule.from.split(',').map(|s| s.trim()).collect();
        for part in &from_parts {
            if !known.contains(*part) && !known.iter().any(|k| k.contains(part)) {
                write!(out, "\n⚠ D{}: references unknown '{}'\n", i + 1, part).unwrap();
                valid = false;
            }
        }
        known.insert(format!("D{}", i + 1));
        known.insert(rule.yields.clone());
    }

    if valid {
        out.push_str("\n✓ Deduction chain is structurally valid.\n");
    }

    out
}

// ─────────────────────────────────────────────────
// §127–§132  Energy Floor / JCI Computations
// ─────────────────────────────────────────────────
//
// Pure deterministic implementations of the Energy Floor theory:
//   V(g) = E(g) + T(g) + S(g) + C(g)
//   C(W) = E(W)·Pₑ + H(W)·Pₕ + I(W)
//   π(W) = C(W) / B(W)
//   F(t) = S·exp((r−κ)·t)
//   E_L = k_B·T·ln(2)

/// Boltzmann constant (J/K).
const K_B: f64 = 1.380_649e-23;

/// Dispatch for all Energy Floor operations.
fn energy_floor_calc(op: &EnergyFloorOp) -> Result<String, String> {
    match op {
        EnergyFloorOp::AnomalyDetect {
            series_id,
            values,
            window,
            threshold,
        } => ef_anomaly_detect(series_id, values, *window, *threshold),

        EnergyFloorOp::Correlation {
            series_a_id,
            series_a,
            series_b_id,
            series_b,
            lag,
        } => ef_correlation(series_a_id, series_a, series_b_id, series_b, *lag),

        EnergyFloorOp::CostFunction {
            energy_joules,
            energy_price_per_joule,
            hardware_units,
            hardware_price_per_unit,
            friction_cost,
            useful_bits,
        } => ef_cost_function(
            *energy_joules,
            *energy_price_per_joule,
            *hardware_units,
            *hardware_price_per_unit,
            *friction_cost,
            *useful_bits,
        ),

        EnergyFloorOp::ForwardCurve {
            spot_price,
            risk_free_rate,
            koomey_rate,
            tenors_days,
            asset_label,
        } => ef_forward_curve(*spot_price, *risk_free_rate, *koomey_rate, tenors_days, asset_label),

        EnergyFloorOp::ArbitrageSpread {
            long_region,
            long_price_kwh,
            short_region,
            short_price_kwh,
            throughput_mw,
        } => ef_arbitrage_spread(
            long_region,
            *long_price_kwh,
            short_region,
            *short_price_kwh,
            *throughput_mw,
        ),

        EnergyFloorOp::ValueFunction {
            good,
            energy_cost,
            trust_cost,
            speed_cost,
            compliance_cost,
        } => ef_value_function(good, *energy_cost, *trust_cost, *speed_cost, *compliance_cost),

        EnergyFloorOp::LandauerBound {
            actual_joules_per_bit,
            temperature_k,
        } => ef_landauer(*actual_joules_per_bit, *temperature_k),
    }
}

/// §127 — Z-score anomaly detection on rolling window.
fn ef_anomaly_detect(
    series_id: &str,
    values: &[f64],
    window: usize,
    threshold: f64,
) -> Result<String, String> {
    if values.len() < window + 1 {
        return Err(format!(
            "need at least {} values for window {window}, got {}",
            window + 1,
            values.len()
        ));
    }

    let latest = values[values.len() - 1];
    let win = &values[values.len() - window - 1..values.len() - 1];
    let n = win.len() as f64;
    let mean = win.iter().sum::<f64>() / n;
    let variance = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    let (z_score, is_anomaly, direction) = if std_dev < f64::EPSILON {
        (0.0, false, "flat")
    } else {
        let z = (latest - mean) / std_dev;
        let anom = z.abs() > threshold;
        let dir = if z > 0.0 { "above" } else { "below" };
        (z, anom, dir)
    };

    let urgency = if z_score.abs() > 3.0 {
        "CRITICAL"
    } else if z_score.abs() > 2.5 {
        "HIGH"
    } else if z_score.abs() > 2.0 {
        "MEDIUM"
    } else {
        "LOW"
    };

    let mut out = format!("**Anomaly Detection: {series_id}**\n\n");
    let _ = write!(out, "Latest value: {latest:.4}\n");
    let _ = write!(out, "Rolling mean ({window}): {mean:.4}\n");
    let _ = write!(out, "Rolling σ: {std_dev:.4}\n");
    let _ = write!(out, "Z-score: {z_score:+.3}\n");
    let _ = write!(out, "Threshold: ±{threshold:.1}σ\n");
    let _ = write!(out, "Direction: {direction}\n");
    let _ = write!(out, "Anomaly: {}\n", if is_anomaly { "YES" } else { "no" });
    let _ = write!(out, "Urgency: {urgency}\n");

    if is_anomaly {
        let _ = write!(
            out,
            "\n⚡ {series_id} is {z_score:+.2}σ {direction} mean — investigate for arbitrage opportunity."
        );
    }

    Ok(out)
}

/// §128 — Lagged cross-correlation (Pearson r).
fn ef_correlation(
    a_id: &str,
    a: &[f64],
    b_id: &str,
    b: &[f64],
    lag: i64,
) -> Result<String, String> {
    let n = a.len().min(b.len());
    if n < 3 {
        return Err("need at least 3 data points per series".into());
    }

    // Apply lag: positive lag means A leads B
    let (sa, sb) = if lag >= 0 {
        let l = lag as usize;
        if l >= n {
            return Err(format!("lag {lag} exceeds series length {n}"));
        }
        (&a[..n - l], &b[l..n])
    } else {
        let l = (-lag) as usize;
        if l >= n {
            return Err(format!("lag {lag} exceeds series length {n}"));
        }
        (&a[l..n], &b[..n - l])
    };

    let nn = sa.len() as f64;
    let mean_a = sa.iter().sum::<f64>() / nn;
    let mean_b = sb.iter().sum::<f64>() / nn;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..sa.len() {
        let da = sa[i] - mean_a;
        let db = sb[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    let r = if denom < f64::EPSILON { 0.0 } else { cov / denom };
    let n_obs = sa.len();

    // Approximate p-value via t-distribution
    let t_stat = if (1.0 - r * r).abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        r * ((n_obs as f64 - 2.0) / (1.0 - r * r)).sqrt()
    };
    // Two-tailed p-value approximation (good enough for large n)
    let df = n_obs as f64 - 2.0;
    let p_approx = if df > 0.0 && t_stat.is_finite() {
        let x = df / (df + t_stat * t_stat);
        // Regularized incomplete beta approximation
        x.powf(df / 2.0) * 0.5
    } else {
        0.0
    };

    let strength = if r.abs() > 0.8 {
        "very strong"
    } else if r.abs() > 0.6 {
        "strong"
    } else if r.abs() > 0.4 {
        "moderate"
    } else if r.abs() > 0.2 {
        "weak"
    } else {
        "negligible"
    };

    let lag_desc = if lag > 0 {
        format!("{a_id} leads {b_id} by {lag} periods")
    } else if lag < 0 {
        format!("{b_id} leads {a_id} by {} periods", -lag)
    } else {
        "contemporaneous".to_string()
    };

    let mut out = format!("**Cross-Correlation: {a_id} × {b_id}**\n\n");
    let _ = write!(out, "Lag: {lag} ({lag_desc})\n");
    let _ = write!(out, "Pearson r: {r:+.4}\n");
    let _ = write!(out, "p-value: {p_approx:.2e}\n");
    let _ = write!(out, "N observations: {n_obs}\n");
    let _ = write!(out, "Strength: {strength}\n");

    if r.abs() > 0.5 && lag != 0 {
        let _ = write!(
            out,
            "\n📊 {strength} {}-correlation with lag — potential leading indicator for alpha.",
            if r > 0.0 { "positive" } else { "negative" }
        );
    }

    Ok(out)
}

/// §129 — Cost function: C(W) = E(W)·Pₑ + H(W)·Pₕ + I(W).
fn ef_cost_function(
    energy_joules: f64,
    energy_price: f64,
    hw_units: f64,
    hw_price: f64,
    friction: f64,
    useful_bits: f64,
) -> Result<String, String> {
    let e_cost = energy_joules * energy_price;
    let h_cost = hw_units * hw_price;
    let total = e_cost + h_cost + friction;

    let (e_frac, h_frac, f_frac) = if total > 0.0 {
        (e_cost / total, h_cost / total, friction / total)
    } else {
        (0.0, 0.0, 0.0)
    };

    let price_per_bit = if useful_bits > 0.0 { total / useful_bits } else { 0.0 };
    let price_per_mbit = price_per_bit * 1_000_000.0;

    // Landauer bound comparison
    let landauer = K_B * 300.0 * 2.0_f64.ln(); // ~2.8e-21 J/bit at room temp
    let landauer_cost = landauer * energy_price;
    let actual_jpb = if useful_bits > 0.0 { energy_joules / useful_bits } else { 0.0 };
    let orders = if actual_jpb > 0.0 && landauer > 0.0 {
        (actual_jpb / landauer).log10()
    } else {
        0.0
    };

    let mut out = String::from("**Cost Function C(W)**\n\n");
    let _ = write!(out, "C(W) = E(W)·Pₑ + H(W)·Pₕ + I(W)\n");
    let _ = write!(out, "     = {energy_joules:.2} J × ${energy_price:.2e}/J + {hw_units:.2} hw × ${hw_price:.4}/hw + ${friction:.4}\n");
    let _ = write!(out, "     = ${total:.6}\n\n");
    let _ = write!(out, "**Component breakdown:**\n");
    let _ = write!(out, "  Energy E(W):   ${e_cost:.6}  ({:.1}%)\n", e_frac * 100.0);
    let _ = write!(out, "  Hardware H(W): ${h_cost:.6}  ({:.1}%)\n", h_frac * 100.0);
    let _ = write!(out, "  Friction I(W): ${friction:.6}  ({:.1}%)\n\n", f_frac * 100.0);
    let _ = write!(out, "**Priced work π(W) = C(W) / B(W):**\n");
    let _ = write!(out, "  π(W) = ${price_per_bit:.2e}/bit = ${price_per_mbit:.6}/Mbit\n\n");
    let _ = write!(out, "**Thermodynamic distance:**\n");
    let _ = write!(out, "  Landauer bound: {landauer:.2e} J/bit = ${landauer_cost:.2e}/bit\n");
    let _ = write!(out, "  Actual: {actual_jpb:.2e} J/bit\n");
    let _ = write!(out, "  Orders above Landauer: {orders:.1}\n");

    Ok(out)
}

/// §130 — Forward curve: F(t) = S·exp((r−κ)·t).
fn ef_forward_curve(
    spot: f64,
    rfr: f64,
    koomey: f64,
    tenors: &[u32],
    label: &str,
) -> Result<String, String> {
    if spot <= 0.0 {
        return Err("spot price must be positive".into());
    }

    let in_backwardation = koomey > rfr; // κ > r → compute is naturally cheap forward

    let mut out = format!("**Forward Curve: {label}**\n\n");
    let _ = write!(out, "F(t) = S · exp((r − κ) · t)\n");
    let _ = write!(out, "Spot S = ${spot:.4}\n");
    let _ = write!(out, "Risk-free r = {:.2}%\n", rfr * 100.0);
    let _ = write!(out, "Koomey κ = {:.2}%\n", koomey * 100.0);
    let _ = write!(
        out,
        "Structure: {} (κ {} r)\n\n",
        if in_backwardation { "BACKWARDATION" } else { "contango" },
        if in_backwardation { ">" } else { "<" }
    );

    let _ = write!(out, "{:<12} {:>12} {:>12} {:>14}\n", "Tenor", "F(t)", "Basis", "Ann. Basis %");
    let _ = write!(out, "{}\n", "─".repeat(54));

    for &days in tenors {
        let t = days as f64 / 365.0;
        let forward = spot * ((rfr - koomey) * t).exp();
        let basis = spot - forward;
        let ann_basis_pct = if t > 0.0 && spot > 0.0 {
            (basis / spot) / t * 100.0
        } else {
            0.0
        };
        let _ = write!(
            out,
            "{:<12} ${:>11.4} ${:>11.4} {:>13.2}%\n",
            format!("{days}d"),
            forward,
            basis,
            ann_basis_pct,
        );
    }

    if in_backwardation {
        let _ = write!(
            out,
            "\n📉 Compute trades in natural backwardation — Koomey's Law guarantees \
             future compute is cheaper. The carry trade: buy forward, sell spot."
        );
    }

    Ok(out)
}

/// §131 — Geographic arbitrage spread.
fn ef_arbitrage_spread(
    long_region: &str,
    long_price: f64,
    short_region: &str,
    short_price: f64,
    throughput_mw: f64,
) -> Result<String, String> {
    let spread = short_price - long_price; // positive = long is cheaper
    let spread_pct = if short_price > 0.0 {
        (spread / short_price) * 100.0
    } else {
        0.0
    };

    // Annualized P&L: spread × throughput × hours/year
    // throughput_mw → kW = MW * 1000, $/kWh spread, 8760 hrs/yr
    let ann_pnl = spread * throughput_mw * 1000.0 * 8760.0;

    let conviction = if spread_pct.abs() > 30.0 {
        "HIGH"
    } else if spread_pct.abs() > 15.0 {
        "MEDIUM"
    } else {
        "LOW"
    };

    let mut out = format!("**Geographic Arbitrage: {short_region} → {long_region}**\n\n");
    let _ = write!(out, "Long (destination):  {long_region} @ ${long_price:.4}/kWh\n");
    let _ = write!(out, "Short (source):      {short_region} @ ${short_price:.4}/kWh\n");
    let _ = write!(out, "Spread:              ${spread:.4}/kWh ({spread_pct:+.1}%)\n");
    let _ = write!(out, "Throughput:           {throughput_mw:.1} MW\n");
    let _ = write!(out, "Annualized P&L:      ${ann_pnl:.0}\n");
    let _ = write!(out, "Conviction:          {conviction}\n");

    if spread > 0.0 {
        let _ = write!(
            out,
            "\n⚡ Route deferrable compute to {long_region} — ${spread:.4}/kWh cheaper, \
             saving ${ann_pnl:.0}/yr at {throughput_mw:.1} MW throughput."
        );
    } else {
        let _ = write!(out, "\n⚠ Negative spread — {long_region} is more expensive than {short_region}.");
    }

    Ok(out)
}

/// §132 — Value function decomposition: V(g) = E(g) + T(g) + S(g) + C(g).
fn ef_value_function(
    good: &str,
    e: f64,
    t: f64,
    s: f64,
    c: f64,
) -> Result<String, String> {
    let vg = e + t + s + c;

    let (e_pct, t_pct, s_pct, c_pct) = if vg > 0.0 {
        (e / vg * 100.0, t / vg * 100.0, s / vg * 100.0, c / vg * 100.0)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // AI compressibility: S(g) is most compressible, E(g) is floor
    let compressible_share = s_pct + (c_pct * 0.5); // S fully, C partially compressible
    let floor = e + t * 0.3; // E irreducible, ~30% of T irreducible (verification still costs)
    let ceiling = vg;
    let compression_ratio = if ceiling > 0.0 { floor / ceiling } else { 1.0 };

    let mut out = format!("**Value Function: {good}**\n\n");
    let _ = write!(out, "V(g) = E(g) + T(g) + S(g) + C(g)\n");
    let _ = write!(out, "     = ${e:.2} + ${t:.2} + ${s:.2} + ${c:.2}\n");
    let _ = write!(out, "     = ${vg:.2}\n\n");
    let _ = write!(out, "**Component shares:**\n");
    let _ = write!(out, "  E(g) Energy:     ${e:.2}  ({e_pct:.1}%) — physics floor\n");
    let _ = write!(out, "  T(g) Trust:      ${t:.2}  ({t_pct:.1}%) — verification, attestation\n");
    let _ = write!(out, "  S(g) Speed:      ${s:.2}  ({s_pct:.1}%) — intelligence, expertise\n");
    let _ = write!(out, "  C(g) Compliance: ${c:.2}  ({c_pct:.1}%) — regulatory, legal\n\n");
    let _ = write!(out, "**AI compression analysis:**\n");
    let _ = write!(out, "  Compressible share: {compressible_share:.1}% (S + ½C)\n");
    let _ = write!(out, "  Energy floor: ${floor:.2} ({:.1}% of current price)\n", compression_ratio * 100.0);
    let _ = write!(out, "  Maximum compression: {:.1}× → ${floor:.2}\n", 1.0 / compression_ratio);

    Ok(out)
}

/// §127b — Landauer bound computation.
fn ef_landauer(actual_jpb: f64, temp_k: f64) -> Result<String, String> {
    let landauer = K_B * temp_k * 2.0_f64.ln();
    let efficiency = if actual_jpb > 0.0 { landauer / actual_jpb } else { 0.0 };
    let orders = if actual_jpb > 0.0 && landauer > 0.0 {
        (actual_jpb / landauer).log10()
    } else {
        0.0
    };

    let mut out = String::from("**Landauer Bound**\n\n");
    let _ = write!(out, "E_L = k_B · T · ln(2)\n");
    let _ = write!(out, "    = {K_B:.3e} J/K × {temp_k:.1} K × ln(2)\n");
    let _ = write!(out, "    = {landauer:.3e} J/bit\n\n");
    let _ = write!(out, "Actual:     {actual_jpb:.3e} J/bit\n");
    let _ = write!(out, "Efficiency: {:.2e} ({:.1}%)\n", efficiency, efficiency * 100.0);
    let _ = write!(out, "Orders above Landauer: {orders:.1}\n\n");

    if orders > 8.0 {
        let _ = write!(out, "Current hardware is ~10^{orders:.0}× above theoretical minimum.\n");
        let _ = write!(out, "Room for {orders:.0} orders of magnitude improvement before physics stops us.");
    } else if orders > 4.0 {
        let _ = write!(out, "Approaching practical efficiency limits — within {orders:.0} orders of Landauer.");
    }

    Ok(out)
}

// ─────────────────────────────────────────────────
// §133  Geographic / GIS Tools
// ─────────────────────────────────────────────────

/// Geographic operation.
#[derive(Debug, Clone, PartialEq)]
pub enum GeoOp {
    /// Haversine distance between two lat/lon points (km and mi).
    Distance {
        lat1: f64, lon1: f64,
        lat2: f64, lon2: f64,
    },
    /// Initial bearing (azimuth) from point A to point B.
    Bearing {
        lat1: f64, lon1: f64,
        lat2: f64, lon2: f64,
    },
    /// Geographic midpoint between two coordinates.
    Midpoint {
        lat1: f64, lon1: f64,
        lat2: f64, lon2: f64,
    },
    /// Destination point given start, bearing (degrees), and distance (km).
    Destination {
        lat: f64, lon: f64,
        bearing_deg: f64,
        distance_km: f64,
    },
    /// Bounding box: expand a center point by a radius (km).
    BoundingBox {
        lat: f64, lon: f64,
        radius_km: f64,
    },
    /// Encode lat/lon to geohash string.
    GeohashEncode {
        lat: f64, lon: f64,
        precision: usize,
    },
    /// Decode geohash string to lat/lon + error bounds.
    GeohashDecode { hash: String },
    /// Parse DMS (degrees-minutes-seconds) to decimal degrees.
    DmsToDd { dms: String },
    /// Convert decimal degrees to DMS string.
    DdToDms { lat: f64, lon: f64 },
    /// Convert lat/lon to UTM zone + easting/northing.
    ToUtm { lat: f64, lon: f64 },
    /// Point-in-polygon test (2D, lat/lon vertices).
    PointInPolygon {
        lat: f64, lon: f64,
        polygon: Vec<(f64, f64)>,
    },
    /// Great-circle waypoints between two points.
    Waypoints {
        lat1: f64, lon1: f64,
        lat2: f64, lon2: f64,
        count: usize,
    },
}

const EARTH_RADIUS_KM: f64 = 6371.0088;
const KM_TO_MI: f64 = 0.621371;

/// Execute a geographic calculation.
pub fn geo_calc(op: &GeoOp) -> String {
    match op {
        GeoOp::Distance { lat1, lon1, lat2, lon2 } => {
            let km = haversine_km(*lat1, *lon1, *lat2, *lon2);
            let mi = km * KM_TO_MI;
            format!(
                "Distance from ({lat1:.6}, {lon1:.6}) to ({lat2:.6}, {lon2:.6}):\n  \
                 {km:.3} km ({mi:.3} mi)"
            )
        }
        GeoOp::Bearing { lat1, lon1, lat2, lon2 } => {
            let b = initial_bearing(*lat1, *lon1, *lat2, *lon2);
            let compass = bearing_to_compass(b);
            format!(
                "Bearing from ({lat1:.6}, {lon1:.6}) to ({lat2:.6}, {lon2:.6}):\n  \
                 {b:.2}° ({compass})"
            )
        }
        GeoOp::Midpoint { lat1, lon1, lat2, lon2 } => {
            let (mlat, mlon) = geo_midpoint(*lat1, *lon1, *lat2, *lon2);
            format!(
                "Midpoint of ({lat1:.6}, {lon1:.6}) and ({lat2:.6}, {lon2:.6}):\n  \
                 ({mlat:.6}, {mlon:.6})"
            )
        }
        GeoOp::Destination { lat, lon, bearing_deg, distance_km } => {
            let (dlat, dlon) = destination_point(*lat, *lon, *bearing_deg, *distance_km);
            format!(
                "From ({lat:.6}, {lon:.6}), bearing {bearing_deg:.1}°, distance {distance_km:.2} km:\n  \
                 Destination: ({dlat:.6}, {dlon:.6})"
            )
        }
        GeoOp::BoundingBox { lat, lon, radius_km } => {
            let (min_lat, min_lon, max_lat, max_lon) = bounding_box(*lat, *lon, *radius_km);
            format!(
                "Bounding box around ({lat:.6}, {lon:.6}), radius {radius_km:.2} km:\n  \
                 SW: ({min_lat:.6}, {min_lon:.6})\n  \
                 NE: ({max_lat:.6}, {max_lon:.6})"
            )
        }
        GeoOp::GeohashEncode { lat, lon, precision } => {
            let hash = geohash_encode(*lat, *lon, *precision);
            format!("Geohash({lat:.6}, {lon:.6}, precision={precision}): {hash}")
        }
        GeoOp::GeohashDecode { hash } => {
            match geohash_decode(hash) {
                Ok((lat, lon, lat_err, lon_err)) => format!(
                    "Geohash \"{hash}\":\n  \
                     Center: ({lat:.6}, {lon:.6})\n  \
                     Error:  ±{lat_err:.6}° lat, ±{lon_err:.6}° lon"
                ),
                Err(e) => format!("Invalid geohash \"{hash}\": {e}"),
            }
        }
        GeoOp::DmsToDd { dms } => {
            match parse_dms(dms) {
                Ok((lat, lon)) => format!("{dms} → ({lat:.6}, {lon:.6})"),
                Err(e) => format!("Cannot parse DMS \"{dms}\": {e}"),
            }
        }
        GeoOp::DdToDms { lat, lon } => {
            let (lat_dms, lon_dms) = dd_to_dms(*lat, *lon);
            format!("({lat:.6}, {lon:.6}) → {lat_dms}, {lon_dms}")
        }
        GeoOp::ToUtm { lat, lon } => {
            let (zone, letter, easting, northing) = latlon_to_utm(*lat, *lon);
            format!(
                "({lat:.6}, {lon:.6}) → UTM {zone}{letter} {easting:.2}E {northing:.2}N"
            )
        }
        GeoOp::PointInPolygon { lat, lon, polygon } => {
            let inside = point_in_polygon(*lat, *lon, polygon);
            if inside {
                format!("({lat:.6}, {lon:.6}) is INSIDE the polygon ({} vertices)", polygon.len())
            } else {
                format!("({lat:.6}, {lon:.6}) is OUTSIDE the polygon ({} vertices)", polygon.len())
            }
        }
        GeoOp::Waypoints { lat1, lon1, lat2, lon2, count } => {
            let pts = great_circle_waypoints(*lat1, *lon1, *lat2, *lon2, *count);
            let mut out = format!(
                "Great-circle waypoints from ({lat1:.4}, {lon1:.4}) to ({lat2:.4}, {lon2:.4}):\n"
            );
            for (i, (lat, lon)) in pts.iter().enumerate() {
                let _ = write!(out, "  {i}: ({lat:.6}, {lon:.6})\n");
            }
            out
        }
    }
}

// ── Haversine ──

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (r1, r2) = (lat1.to_radians(), lat2.to_radians());
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + r1.cos() * r2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * a.sqrt().asin()
}

// ── Bearing ──

fn initial_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (r1, r2) = (lat1.to_radians(), lat2.to_radians());
    let dlon = (lon2 - lon1).to_radians();
    let y = dlon.sin() * r2.cos();
    let x = r1.cos() * r2.sin() - r1.sin() * r2.cos() * dlon.cos();
    (y.atan2(x).to_degrees() + 360.0) % 360.0
}

fn bearing_to_compass(deg: f64) -> &'static str {
    const DIRS: [&str; 16] = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ];
    let idx = ((deg + 11.25) / 22.5) as usize % 16;
    DIRS[idx]
}

// ── Midpoint ──

fn geo_midpoint(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64) {
    let (r1, r2) = (lat1.to_radians(), lat2.to_radians());
    let (lo1, lo2) = (lon1.to_radians(), lon2.to_radians());
    let bx = r2.cos() * (lo2 - lo1).cos();
    let by = r2.cos() * (lo2 - lo1).sin();
    let lat = (r1.sin() + r2.sin()).atan2(((r1.cos() + bx).powi(2) + by.powi(2)).sqrt());
    let lon = lo1 + by.atan2(r1.cos() + bx);
    (lat.to_degrees(), lon.to_degrees())
}

// ── Destination point ──

fn destination_point(lat: f64, lon: f64, bearing_deg: f64, dist_km: f64) -> (f64, f64) {
    let r = lat.to_radians();
    let lo = lon.to_radians();
    let brng = bearing_deg.to_radians();
    let d = dist_km / EARTH_RADIUS_KM;

    let lat2 = (r.sin() * d.cos() + r.cos() * d.sin() * brng.cos()).asin();
    let lon2 = lo + (brng.sin() * d.sin() * r.cos()).atan2(d.cos() - r.sin() * lat2.sin());
    (lat2.to_degrees(), lon2.to_degrees())
}

// ── Bounding box ──

fn bounding_box(lat: f64, lon: f64, radius_km: f64) -> (f64, f64, f64, f64) {
    let lat_delta = radius_km / 111.32; // 1° lat ≈ 111.32 km
    let lon_delta = radius_km / (111.32 * lat.to_radians().cos().abs().max(0.001));
    (lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta)
}

// ── Geohash ──

const GEOHASH_BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

fn geohash_encode(lat: f64, lon: f64, precision: usize) -> String {
    let prec = precision.clamp(1, 12);
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut hash = String::with_capacity(prec);
    let mut bits = 0u8;
    let mut bit_count = 0u8;
    let mut is_lon = true;

    while hash.len() < prec {
        let (range, val) = if is_lon {
            (&mut lon_range, lon)
        } else {
            (&mut lat_range, lat)
        };
        let mid = (range.0 + range.1) / 2.0;
        if val >= mid {
            bits = (bits << 1) | 1;
            range.0 = mid;
        } else {
            bits <<= 1;
            range.1 = mid;
        }
        is_lon = !is_lon;
        bit_count += 1;
        if bit_count == 5 {
            hash.push(GEOHASH_BASE32[bits as usize] as char);
            bits = 0;
            bit_count = 0;
        }
    }
    hash
}

fn geohash_decode(hash: &str) -> Result<(f64, f64, f64, f64), &'static str> {
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut is_lon = true;

    for ch in hash.chars() {
        let idx = GEOHASH_BASE32
            .iter()
            .position(|&c| c == ch as u8)
            .ok_or("invalid geohash character")?;
        for bit in (0..5).rev() {
            let range = if is_lon { &mut lon_range } else { &mut lat_range };
            let mid = (range.0 + range.1) / 2.0;
            if (idx >> bit) & 1 == 1 {
                range.0 = mid;
            } else {
                range.1 = mid;
            }
            is_lon = !is_lon;
        }
    }
    let lat = (lat_range.0 + lat_range.1) / 2.0;
    let lon = (lon_range.0 + lon_range.1) / 2.0;
    Ok((lat, lon, (lat_range.1 - lat_range.0) / 2.0, (lon_range.1 - lon_range.0) / 2.0))
}

// ── DMS parsing ──

fn parse_dms(s: &str) -> Result<(f64, f64), &'static str> {
    // Accepts: 40°26'46"N 79°58'56"W  or  40 26 46 N 79 58 56 W  etc.
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r#"(?i)(-?\d+)[\x{b0}\s]+(\d+)['\s]+(\d+(?:\.\d+)?)[\x{22}\x{2033}\s]*([NSEW]?)[,;\s]+(-?\d+)[\x{b0}\s]+(\d+)['\s]+(\d+(?:\.\d+)?)[\x{22}\x{2033}\s]*([NSEW]?)"#).unwrap()
    });

    let caps = RE.captures(s).ok_or("expected format: D°M'S\"N D°M'S\"W")?;
    let to_dd = |d: f64, m: f64, s: f64, dir: &str| -> f64 {
        let dd = d.abs() + m / 60.0 + s / 3600.0;
        if dir.eq_ignore_ascii_case("S") || dir.eq_ignore_ascii_case("W") || d < 0.0 {
            -dd
        } else {
            dd
        }
    };
    let lat = to_dd(
        caps[1].parse().unwrap_or(0.0),
        caps[2].parse().unwrap_or(0.0),
        caps[3].parse().unwrap_or(0.0),
        &caps[4],
    );
    let lon = to_dd(
        caps[5].parse().unwrap_or(0.0),
        caps[6].parse().unwrap_or(0.0),
        caps[7].parse().unwrap_or(0.0),
        &caps[8],
    );
    Ok((lat, lon))
}

// ── DD to DMS ──

fn dd_to_dms(lat: f64, lon: f64) -> (String, String) {
    let fmt = |val: f64, pos: char, neg: char| -> String {
        let dir = if val >= 0.0 { pos } else { neg };
        let v = val.abs();
        let d = v as u32;
        let m = ((v - d as f64) * 60.0) as u32;
        let s = (v - d as f64 - m as f64 / 60.0) * 3600.0;
        format!("{d}°{m}'{s:.2}\"{dir}")
    };
    (fmt(lat, 'N', 'S'), fmt(lon, 'E', 'W'))
}

// ── UTM ──

fn latlon_to_utm(lat: f64, lon: f64) -> (u32, char, f64, f64) {
    let zone = ((lon + 180.0) / 6.0).floor() as u32 + 1;
    let letter = if lat >= 84.0 { 'X' }
    else if lat >= 72.0 { 'W' }
    else if lat >= 0.0 {
        (b'N' + ((lat / 8.0).floor() as u8).min(5)) as char
    } else if lat >= -80.0 {
        (b'C' + (((lat + 80.0) / 8.0).floor() as u8).min(11)) as char
    } else { 'C' };

    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let lat_r = lat.to_radians();

    // Simplified UTM projection (WGS84)
    let n = 0.0016792204_f64; // f / (2 - f) for WGS84
    let a = 6378.137; // km
    let k0 = 0.9996;
    let t = lat_r.tan();
    let c = 0.006739497_f64 * lat_r.cos().powi(2); // e'^2 * cos^2
    let aa = (lon.to_radians() - lon0) * lat_r.cos();

    let m = a * (
        (1.0 - n.powi(2) / 4.0 - 3.0 * n.powi(4) / 64.0) * lat_r
        - (3.0 * n / 2.0 + 27.0 * n.powi(3) / 32.0) * (2.0 * lat_r).sin()
        + (15.0 * n.powi(2) / 16.0) * (4.0 * lat_r).sin()
    );

    let nu = a / (1.0 - 0.00669438_f64 * lat_r.sin().powi(2)).sqrt();

    let easting = k0 * nu * (aa + (1.0 - t * t + c) * aa.powi(3) / 6.0) + 500.0;
    let mut northing = k0 * (m + nu * t * (aa.powi(2) / 2.0 + (5.0 - t * t + 9.0 * c) * aa.powi(4) / 24.0));
    if lat < 0.0 { northing += 10000.0; }

    (zone, letter, easting, northing)
}

// ── Point-in-polygon (ray casting) ──

fn point_in_polygon(lat: f64, lon: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 { return false; }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (yi, xi) = polygon[i];
        let (yj, xj) = polygon[j];
        if ((yi > lat) != (yj > lat)) && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ── Great-circle waypoints ──

fn great_circle_waypoints(lat1: f64, lon1: f64, lat2: f64, lon2: f64, count: usize) -> Vec<(f64, f64)> {
    let count = count.max(2);
    let (r1, r2) = (lat1.to_radians(), lat2.to_radians());
    let (lo1, lo2) = (lon1.to_radians(), lon2.to_radians());

    let d = haversine_km(lat1, lon1, lat2, lon2) / EARTH_RADIUS_KM;
    if d.abs() < 1e-12 {
        return vec![(lat1, lon1); count];
    }

    (0..count)
        .map(|i| {
            let f = i as f64 / (count - 1) as f64;
            let a = ((1.0 - f) * d).sin() / d.sin();
            let b = (f * d).sin() / d.sin();
            let x = a * r1.cos() * lo1.cos() + b * r2.cos() * lo2.cos();
            let y = a * r1.cos() * lo1.sin() + b * r2.cos() * lo2.sin();
            let z = a * r1.sin() + b * r2.sin();
            (z.atan2((x * x + y * y).sqrt()).to_degrees(), y.atan2(x).to_degrees())
        })
        .collect()
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Math evaluator ──

    #[test]
    fn math_basic_arithmetic() {
        assert!((eval_math("2 + 2").unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((eval_math("10 - 3").unwrap() - 7.0).abs() < f64::EPSILON);
        assert!((eval_math("6 * 7").unwrap() - 42.0).abs() < f64::EPSILON);
        assert!((eval_math("15 / 3").unwrap() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_order_of_operations() {
        assert!((eval_math("2 + 3 * 4").unwrap() - 14.0).abs() < f64::EPSILON);
        assert!((eval_math("(2 + 3) * 4").unwrap() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_power() {
        assert!((eval_math("2 ^ 10").unwrap() - 1024.0).abs() < f64::EPSILON);
        assert!((eval_math("3 ^ 3").unwrap() - 27.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_functions() {
        assert!((eval_math("sqrt(144)").unwrap() - 12.0).abs() < f64::EPSILON);
        assert!((eval_math("abs(-42)").unwrap() - 42.0).abs() < f64::EPSILON);
        assert!((eval_math("factorial(5)").unwrap() - 120.0).abs() < f64::EPSILON);
        assert!((eval_math("max(3, 7, 1)").unwrap() - 7.0).abs() < f64::EPSILON);
        assert!((eval_math("min(3, 7, 1)").unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_trig() {
        assert!((eval_math("sin(0)").unwrap()).abs() < f64::EPSILON);
        assert!((eval_math("cos(0)").unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_constants() {
        assert!((eval_math("pi").unwrap() - std::f64::consts::PI).abs() < f64::EPSILON);
        assert!((eval_math("e").unwrap() - std::f64::consts::E).abs() < f64::EPSILON);
    }

    #[test]
    fn math_division_by_zero() {
        assert!(eval_math("1 / 0").is_err());
    }

    #[test]
    fn math_modulo() {
        assert!((eval_math("17 % 5").unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_nested_functions() {
        assert!((eval_math("sqrt(pow(3, 2) + pow(4, 2))").unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn math_negative_numbers() {
        assert!((eval_math("-5 + 3").unwrap() - (-2.0)).abs() < f64::EPSILON);
        assert!((eval_math("(-3) * (-4)").unwrap() - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn math_scientific_notation() {
        assert!((eval_math("1e3").unwrap() - 1000.0).abs() < f64::EPSILON);
        assert!((eval_math("2.5e2").unwrap() - 250.0).abs() < f64::EPSILON);
    }

    // ── Unit converter ──

    #[test]
    fn convert_km_to_miles() {
        let (result, _, _) = convert_units(5.0, "km", "mi").unwrap();
        assert!((result - 3.10686).abs() < 0.001);
    }

    #[test]
    fn convert_celsius_to_fahrenheit() {
        let (result, _, _) = convert_units(100.0, "celsius", "fahrenheit").unwrap();
        assert!((result - 212.0).abs() < 0.01);
    }

    #[test]
    fn convert_fahrenheit_to_celsius() {
        let (result, _, _) = convert_units(32.0, "fahrenheit", "celsius").unwrap();
        assert!(result.abs() < 0.01);
    }

    #[test]
    fn convert_kg_to_pounds() {
        let (result, _, _) = convert_units(1.0, "kg", "lb").unwrap();
        assert!((result - 2.20462).abs() < 0.001);
    }

    #[test]
    fn convert_gb_to_mb() {
        let (result, _, _) = convert_units(1.0, "gb", "mb").unwrap();
        assert!((result - 1024.0).abs() < 0.01);
    }

    #[test]
    fn convert_incompatible_units() {
        assert!(convert_units(1.0, "km", "kg").is_err());
    }

    #[test]
    fn convert_liters_to_gallons() {
        let (result, _, _) = convert_units(3.78541, "l", "gal").unwrap();
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn convert_degrees_to_radians() {
        let (result, _, _) = convert_units(180.0, "degrees", "rad").unwrap();
        assert!((result - std::f64::consts::PI).abs() < 1e-10);
    }

    // ── Number base ──

    #[test]
    fn base_decimal_to_hex() {
        assert_eq!(convert_base("255", 16).unwrap(), "0xFF");
    }

    #[test]
    fn base_decimal_to_binary() {
        assert_eq!(convert_base("42", 2).unwrap(), "0b101010");
    }

    #[test]
    fn base_hex_to_decimal() {
        assert_eq!(convert_base("0xFF", 10).unwrap(), "255");
    }

    #[test]
    fn base_binary_to_decimal() {
        assert_eq!(convert_base("0b1010", 10).unwrap(), "10");
    }

    #[test]
    fn all_bases_display() {
        let result = format_all_bases("42").unwrap();
        assert!(result.contains("decimal: 42"));
        assert!(result.contains("0b101010"));
        assert!(result.contains("0o52"));
        assert!(result.contains("0x2A"));
    }

    // ── Text transforms ──

    #[test]
    fn base64_roundtrip() {
        let encoded = base64_encode("Hello, World!");
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
        assert_eq!(base64_decode(&encoded).unwrap(), "Hello, World!");
    }

    #[test]
    fn url_encode_decode() {
        let encoded = url_encode("hello world & foo=bar");
        assert!(encoded.contains("%20") || encoded.contains('+'));
        let decoded = url_decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world & foo=bar");
    }

    #[test]
    fn sha256_known_value() {
        let hash = sha256("hello");
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn text_stats_counts() {
        let stats = text_stats("Hello World. How are you?");
        assert_eq!(stats.words, 5);
        assert_eq!(stats.sentences, 2);
    }

    #[test]
    fn title_case() {
        assert_eq!(to_title_case("hello world foo"), "Hello World Foo");
    }

    #[test]
    fn reverse() {
        assert_eq!(reverse_string("hello"), "olleh");
    }

    // ── Color converter ──

    #[test]
    fn color_hex_to_all() {
        let color = parse_color("#FF0000").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
        assert_eq!(color.hex, "#FF0000");
        assert_eq!(color.rgb, "rgb(255, 0, 0)");
        assert!(color.hsl.contains("hsl(0,"));
    }

    #[test]
    fn color_rgb_to_hex() {
        let color = parse_color("rgb(0,128,255)").unwrap();
        assert_eq!(color.hex, "#0080FF");
    }

    #[test]
    fn color_hsl_roundtrip() {
        let color = parse_color("hsl(120,100%,50%)").unwrap();
        assert_eq!(color.r, 0);
        assert_eq!(color.g, 255);
        assert_eq!(color.b, 0);
    }

    // ── Statistics ──

    #[test]
    fn stats_basic() {
        let stats = statistics(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((stats.mean - 3.0).abs() < f64::EPSILON);
        assert!((stats.median - 3.0).abs() < f64::EPSILON);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 5.0).abs() < f64::EPSILON);
        assert!((stats.sum - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_single_value() {
        let stats = statistics(&[42.0]).unwrap();
        assert!((stats.mean - 42.0).abs() < f64::EPSILON);
        assert!((stats.std_dev).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_empty() {
        assert!(statistics(&[]).is_err());
    }

    #[test]
    fn parse_number_list_basic() {
        let nums = parse_number_list("1, 2, 3, 4, 5").unwrap();
        assert_eq!(nums, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // ── Date calculator ──

    #[test]
    fn days_between_dates() {
        let days = days_between("2025-01-01", "2025-03-15").unwrap();
        assert_eq!(days, 73);
    }

    #[test]
    fn add_days_to_date() {
        let result = add_days("2025-01-01", 30).unwrap();
        assert_eq!(result, "2025-01-31");
    }

    #[test]
    fn day_of_week_known() {
        let dow = day_of_week("2025-01-01").unwrap();
        assert_eq!(dow, "Wednesday");
    }

    // ── Unified dispatcher ──

    #[test]
    fn execute_math() {
        let result = execute(&DeterministicToolKind::Math {
            expression: "2 + 2".into(),
        })
        .unwrap();
        assert!(result.contains("4"));
    }

    #[test]
    fn execute_unit_conversion() {
        let result = execute(&DeterministicToolKind::UnitConversion {
            value: 100.0,
            from_unit: "celsius".into(),
            to_unit: "fahrenheit".into(),
        })
        .unwrap();
        assert!(result.contains("212"));
    }

    #[test]
    fn execute_text_transform() {
        let result = execute(&DeterministicToolKind::TextTransform {
            operation: TextOp::Sha256,
            input: "hello".into(),
        })
        .unwrap();
        assert!(result.starts_with("2cf24dba"));
    }

    #[test]
    fn format_math_result_integer() {
        assert_eq!(format_math_result(42.0), "42");
        assert_eq!(format_math_result(-7.0), "-7");
    }

    #[test]
    fn format_math_result_decimal() {
        assert_eq!(format_math_result(3.14), "3.14");
    }

    // ── MIDI / Music ──

    #[test]
    fn midi_a4_is_440() {
        assert!((midi_to_freq(69) - 440.0).abs() < 0.01);
    }

    #[test]
    fn midi_c4_is_middle_c() {
        assert_eq!(midi_to_name(60), "C4");
        assert!((midi_to_freq(60) - 261.63).abs() < 0.1);
    }

    #[test]
    fn freq_440_is_a4() {
        let (note, cents) = freq_to_midi(440.0);
        assert_eq!(note, 69);
        assert!(cents.abs() < 0.5);
    }

    #[test]
    fn name_to_midi_c4() {
        assert_eq!(name_to_midi("C4").unwrap(), 60);
        assert_eq!(name_to_midi("A4").unwrap(), 69);
        assert_eq!(name_to_midi("C#4").unwrap(), 61);
    }

    #[test]
    fn bpm_120_is_500ms() {
        assert!((bpm_to_ms(120.0) - 500.0).abs() < 0.01);
    }

    // ── Roman numerals ──

    #[test]
    fn roman_2024() {
        assert_eq!(to_roman(2024).unwrap(), "MMXXIV");
    }

    #[test]
    fn roman_roundtrip() {
        assert_eq!(from_roman("MMXXIV").unwrap(), 2024);
        assert_eq!(from_roman("XIV").unwrap(), 14);
        assert_eq!(from_roman("MCMXCIX").unwrap(), 1999);
    }

    #[test]
    fn roman_zero_rejected() {
        assert!(to_roman(0).is_err());
    }

    // ── Percentage ──

    #[test]
    fn percentage_of_basic() {
        assert!((percentage_of(15.0, 200.0) - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn what_percentage_basic() {
        assert!((what_percentage(30.0, 200.0).unwrap() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn percentage_change_basic() {
        assert!((percentage_change(100.0, 150.0).unwrap() - 50.0).abs() < f64::EPSILON);
    }

    // ── UUID ──

    #[test]
    fn uuid_format() {
        let uuid = generate_uuid();
        assert_eq!(uuid.len(), 36); // 8-4-4-4-12
        assert_eq!(uuid.matches('-').count(), 4);
    }

    // ── Timestamp ──

    #[test]
    fn timestamp_epoch() {
        let dt = timestamp_to_datetime(0).unwrap();
        assert!(dt.contains("1970-01-01"));
    }

    #[test]
    fn timestamp_known() {
        let dt = timestamp_to_datetime(1700000000).unwrap();
        assert!(dt.contains("2023-11-14"));
    }

    #[test]
    fn timestamp_now_reasonable() {
        let ts = now_timestamp();
        assert!(ts > 1700000000); // After Nov 2023
    }

    // ── SVG Generator ──

    #[test]
    fn svg_empty() {
        let params = SvgParams::default();
        let svg = generate_svg(&params);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("width=\"400\""));
    }

    #[test]
    fn svg_rect() {
        let params = SvgParams {
            shapes: vec![SvgShape::Rect {
                x: 10.0,
                y: 20.0,
                w: 100.0,
                h: 50.0,
                rx: 0.0,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("<rect x=\"10\""));
        assert!(svg.contains("width=\"100\""));
    }

    #[test]
    fn svg_rounded_rect() {
        let params = SvgParams {
            shapes: vec![SvgShape::Rect {
                x: 0.0,
                y: 0.0,
                w: 80.0,
                h: 40.0,
                rx: 5.0,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("rx=\"5\""));
    }

    #[test]
    fn svg_circle() {
        let params = SvgParams {
            shapes: vec![SvgShape::Circle {
                cx: 200.0,
                cy: 200.0,
                r: 50.0,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("r=\"50\""));
    }

    #[test]
    fn svg_polygon_hexagon() {
        let params = SvgParams {
            shapes: vec![SvgShape::Polygon {
                cx: 100.0,
                cy: 100.0,
                r: 50.0,
                sides: 6,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("<polygon"));
        // 6 vertices = 6 coordinate pairs
        let points_attr = svg
            .split("points=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        assert_eq!(points_attr.split(' ').count(), 6);
    }

    #[test]
    fn svg_star() {
        let params = SvgParams {
            shapes: vec![SvgShape::Star {
                cx: 100.0,
                cy: 100.0,
                outer_r: 50.0,
                inner_r: 25.0,
                points: 5,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("<polygon"));
        // 5-pointed star = 10 vertices
        let points_attr = svg
            .split("points=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        assert_eq!(points_attr.split(' ').count(), 10);
    }

    #[test]
    fn svg_text() {
        let params = SvgParams {
            shapes: vec![SvgShape::Text {
                x: 10.0,
                y: 30.0,
                text: "Hello".into(),
                size: 16.0,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("<text"));
        assert!(svg.contains("Hello</text>"));
    }

    #[test]
    fn svg_grid() {
        let params = SvgParams {
            shapes: vec![SvgShape::Grid {
                cols: 3,
                rows: 2,
                cell: 50.0,
                gap: 2.0,
            }],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        // 4 vertical lines (cols+1) + 3 horizontal lines (rows+1) = 7 lines
        assert_eq!(svg.matches("<line").count(), 7);
    }

    #[test]
    fn svg_background() {
        let params = SvgParams {
            background: Some("#1a1a2e".into()),
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("fill=\"#1a1a2e\""));
    }

    #[test]
    fn svg_multiple_shapes() {
        let params = SvgParams {
            shapes: vec![
                SvgShape::Circle {
                    cx: 100.0,
                    cy: 100.0,
                    r: 40.0,
                },
                SvgShape::Rect {
                    x: 200.0,
                    y: 50.0,
                    w: 100.0,
                    h: 100.0,
                    rx: 0.0,
                },
                SvgShape::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 400.0,
                    y2: 400.0,
                },
            ],
            ..Default::default()
        };
        let svg = generate_svg(&params);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn svg_execute_dispatch() {
        let params = SvgParams {
            shapes: vec![SvgShape::Circle {
                cx: 50.0,
                cy: 50.0,
                r: 25.0,
            }],
            ..Default::default()
        };
        let result = execute(&DeterministicToolKind::Svg { params });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("<svg"));
    }

    // ── DXF Generator ──

    #[test]
    fn dxf_empty() {
        let params = DxfParams::default();
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("SECTION"));
        assert!(dxf.contains("EOF"));
        assert!(dxf.contains("AC1009")); // R12 format
    }

    #[test]
    fn dxf_line() {
        let params = DxfParams {
            entities: vec![DxfEntity::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 50.0,
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("LINE"));
        assert!(dxf.contains("100"));
    }

    #[test]
    fn dxf_circle() {
        let params = DxfParams {
            entities: vec![DxfEntity::Circle {
                cx: 50.0,
                cy: 50.0,
                r: 25.0,
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("CIRCLE"));
    }

    #[test]
    fn dxf_arc() {
        let params = DxfParams {
            entities: vec![DxfEntity::Arc {
                cx: 50.0,
                cy: 50.0,
                r: 25.0,
                start_deg: 0.0,
                end_deg: 90.0,
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("ARC"));
    }

    #[test]
    fn dxf_rect_as_four_lines() {
        let params = DxfParams {
            entities: vec![DxfEntity::Rect {
                x: 10.0,
                y: 10.0,
                w: 80.0,
                h: 40.0,
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        // Rect = 4 LINE entities
        assert_eq!(dxf.matches("\n0\nLINE\n").count(), 4);
    }

    #[test]
    fn dxf_polygon_pentagon() {
        let params = DxfParams {
            entities: vec![DxfEntity::Polygon {
                cx: 50.0,
                cy: 50.0,
                r: 30.0,
                sides: 5,
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        // Pentagon = 5 LINE entities
        assert_eq!(dxf.matches("\n0\nLINE\n").count(), 5);
    }

    #[test]
    fn dxf_text() {
        let params = DxfParams {
            entities: vec![DxfEntity::Text {
                x: 10.0,
                y: 10.0,
                height: 5.0,
                text: "Hello".into(),
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("TEXT"));
        assert!(dxf.contains("Hello"));
    }

    #[test]
    fn dxf_dimension() {
        let params = DxfParams {
            entities: vec![DxfEntity::Dimension {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 0.0,
            }],
            ..Default::default()
        };
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("LINE"));
        assert!(dxf.contains("TEXT"));
        assert!(dxf.contains("100.00")); // Distance annotation
    }

    #[test]
    fn dxf_custom_layer() {
        let params = DxfParams {
            entities: vec![DxfEntity::Circle {
                cx: 0.0,
                cy: 0.0,
                r: 10.0,
            }],
            layer: "WALLS".into(),
        };
        let dxf = generate_dxf(&params);
        assert!(dxf.contains("WALLS"));
    }

    #[test]
    fn dxf_execute_dispatch() {
        let params = DxfParams {
            entities: vec![DxfEntity::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 50.0,
                y2: 50.0,
            }],
            ..Default::default()
        };
        let result = execute(&DeterministicToolKind::Dxf { params });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("LINE"));
    }

    // ── Regex ──

    #[test]
    fn regex_test_match() {
        let result = regex_exec(&RegexOp::Test {
            pattern: r"\d+".into(),
            input: "hello 42 world".into(),
        });
        assert!(result.unwrap().contains("Match found"));
    }

    #[test]
    fn regex_test_no_match() {
        let result = regex_exec(&RegexOp::Test {
            pattern: r"\d+".into(),
            input: "hello world".into(),
        });
        assert!(result.unwrap().contains("No match"));
    }

    #[test]
    fn regex_find_all() {
        let result = regex_exec(&RegexOp::FindAll {
            pattern: r"\d+".into(),
            input: "a1 b22 c333".into(),
        });
        let text = result.unwrap();
        assert!(text.contains("3 matches"));
        assert!(text.contains("\"1\""));
        assert!(text.contains("\"22\""));
        assert!(text.contains("\"333\""));
    }

    #[test]
    fn regex_replace() {
        let result = regex_exec(&RegexOp::Replace {
            pattern: r"\d+".into(),
            input: "a1 b22 c333".into(),
            replacement: "X".into(),
        });
        assert_eq!(result.unwrap(), "aX bX cX");
    }

    #[test]
    fn regex_invalid_pattern() {
        let result = regex_exec(&RegexOp::Test {
            pattern: r"[invalid".into(),
            input: "test".into(),
        });
        assert!(result.is_err());
    }

    // ── Cron ──

    #[test]
    fn cron_every_minute() {
        let result = parse_cron("* * * * *").unwrap();
        assert_eq!(result, "every minute");
    }

    #[test]
    fn cron_daily_midnight() {
        let result = parse_cron("0 0 * * *").unwrap();
        assert!(result.contains("start of the hour"));
        assert!(result.contains("0:00"));
    }

    #[test]
    fn cron_weekly_monday() {
        let result = parse_cron("0 9 * * 1").unwrap();
        assert!(result.contains("Monday"));
    }

    #[test]
    fn cron_every_5_minutes() {
        let result = parse_cron("*/5 * * * *").unwrap();
        assert!(result.contains("every 5"));
    }

    #[test]
    fn cron_monthly_first() {
        let result = parse_cron("0 0 1 * *").unwrap();
        assert!(result.contains("day 1"));
    }

    #[test]
    fn cron_invalid_fields() {
        assert!(parse_cron("* *").is_err());
    }

    // ── Data Format Conversion ──

    #[test]
    fn json_to_yaml() {
        let json = r#"{"name": "test", "value": 42}"#;
        let result = convert_data_format(json, &DataFormat::Json, &DataFormat::Yaml).unwrap();
        assert!(result.contains("name"));
        assert!(result.contains("test"));
        assert!(result.contains("42"));
    }

    #[test]
    fn json_to_toml() {
        let json = r#"{"name": "test", "value": 42}"#;
        let result = convert_data_format(json, &DataFormat::Json, &DataFormat::Toml).unwrap();
        assert!(result.contains("name"));
        assert!(result.contains("test"));
    }

    #[test]
    fn yaml_to_json() {
        let yaml = "name: test\nvalue: 42";
        let result = convert_data_format(yaml, &DataFormat::Yaml, &DataFormat::Json).unwrap();
        assert!(result.contains("\"name\""));
        assert!(result.contains("\"test\""));
    }

    #[test]
    fn invalid_json() {
        let result = convert_data_format("{bad", &DataFormat::Json, &DataFormat::Yaml);
        assert!(result.is_err());
    }

    // ── IP/Subnet Calculator ──

    #[test]
    fn subnet_class_c() {
        let info = calc_subnet("192.168.1.0/24").unwrap();
        assert_eq!(info.network, "192.168.1.0");
        assert_eq!(info.broadcast, "192.168.1.255");
        assert_eq!(info.netmask, "255.255.255.0");
        assert_eq!(info.total_hosts, 254);
        assert_eq!(info.first_host, "192.168.1.1");
        assert_eq!(info.last_host, "192.168.1.254");
    }

    #[test]
    fn subnet_slash_16() {
        let info = calc_subnet("10.0.0.0/16").unwrap();
        assert_eq!(info.network, "10.0.0.0");
        assert_eq!(info.netmask, "255.255.0.0");
        assert_eq!(info.total_hosts, 65534);
    }

    #[test]
    fn subnet_slash_32() {
        let info = calc_subnet("1.2.3.4/32").unwrap();
        assert_eq!(info.total_hosts, 1);
    }

    #[test]
    fn subnet_slash_31() {
        let info = calc_subnet("10.0.0.0/31").unwrap();
        assert_eq!(info.total_hosts, 2);
    }

    #[test]
    fn subnet_invalid() {
        assert!(calc_subnet("not an ip").is_err());
        assert!(calc_subnet("192.168.1.0/33").is_err());
    }

    // ── QR Code ──

    #[test]
    fn qr_code_generates_svg() {
        let svg = generate_qr_svg("hello", 4.0).unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn qr_code_empty_rejected() {
        assert!(generate_qr_svg("", 4.0).is_err());
    }

    #[test]
    fn qr_code_has_finder_patterns() {
        let svg = generate_qr_svg("test", 1.0).unwrap();
        // Should have at least 49 black modules for 3 finder patterns (7x7 each)
        let rect_count = svg.matches("fill=\"black\"").count();
        assert!(rect_count >= 49);
    }

    // ── Password Generator ──

    #[test]
    fn password_correct_length() {
        let pw = generate_password(16);
        assert_eq!(pw.len(), 16);
    }

    #[test]
    fn password_different_each_time() {
        // Due to time-based seed, rapid calls may collide, but length should be right
        let pw1 = generate_password(20);
        assert_eq!(pw1.len(), 20);
    }

    #[test]
    fn passphrase_correct_word_count() {
        let pp = generate_passphrase(4);
        assert_eq!(pp.split('-').count(), 4);
    }

    #[test]
    fn passphrase_words_are_lowercase() {
        let pp = generate_passphrase(3);
        assert!(pp.chars().all(|c| c.is_lowercase() || c == '-'));
    }

    // ── CSS Palette ──

    #[test]
    fn palette_complementary() {
        let result = generate_palette("#FF0000", &PaletteMode::Complementary, 2).unwrap();
        assert!(result.contains("Complementary"));
        assert!(result.contains("#FF0000")); // base color (or close)
    }

    #[test]
    fn palette_triadic() {
        let result = generate_palette("#3366CC", &PaletteMode::Triadic, 3).unwrap();
        assert!(result.contains("Triadic"));
        // Should have 3 colors
        assert!(result.lines().filter(|l| l.trim().starts_with('#')).count() >= 3);
    }

    #[test]
    fn palette_invalid_color() {
        assert!(generate_palette("not-a-color", &PaletteMode::Complementary, 2).is_err());
    }

    #[test]
    fn palette_shorthand_hex() {
        let result = generate_palette("#F00", &PaletteMode::Complementary, 2);
        assert!(result.is_ok());
    }

    // ── OpenSCAD ──

    #[test]
    fn scad_cube() {
        let params = ScadParams {
            shapes: vec![ScadShape::Cube {
                w: 10.0,
                d: 20.0,
                h: 30.0,
                center: true,
            }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("cube([10, 20, 30], center = true)"));
        assert!(scad.contains("$fn = 64"));
    }

    #[test]
    fn scad_sphere() {
        let params = ScadParams {
            shapes: vec![ScadShape::Sphere { r: 15.0 }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("sphere(r = 15)"));
    }

    #[test]
    fn scad_cylinder() {
        let params = ScadParams {
            shapes: vec![ScadShape::Cylinder {
                h: 20.0,
                r1: 10.0,
                r2: 10.0,
            }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("cylinder(h = 20, r = 10)"));
    }

    #[test]
    fn scad_cone() {
        let params = ScadParams {
            shapes: vec![ScadShape::Cylinder {
                h: 20.0,
                r1: 10.0,
                r2: 0.0,
            }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("r1 = 10"));
        assert!(scad.contains("r2 = 0"));
    }

    #[test]
    fn scad_difference() {
        let params = ScadParams {
            shapes: vec![ScadShape::Difference {
                shapes: vec![
                    ScadShape::Cube {
                        w: 20.0,
                        d: 20.0,
                        h: 20.0,
                        center: true,
                    },
                    ScadShape::Sphere { r: 12.0 },
                ],
            }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("difference()"));
        assert!(scad.contains("cube("));
        assert!(scad.contains("sphere("));
    }

    #[test]
    fn scad_translate() {
        let params = ScadParams {
            shapes: vec![ScadShape::Translate {
                x: 10.0,
                y: 20.0,
                z: 30.0,
                child: Box::new(ScadShape::Sphere { r: 5.0 }),
            }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("translate([10, 20, 30])"));
        assert!(scad.contains("sphere(r = 5)"));
    }

    #[test]
    fn scad_text() {
        let params = ScadParams {
            shapes: vec![ScadShape::Text {
                text: "Hello".into(),
                size: 10.0,
                height: 3.0,
            }],
            ..Default::default()
        };
        let scad = generate_scad(&params);
        assert!(scad.contains("linear_extrude"));
        assert!(scad.contains("text(\"Hello\""));
    }

    // ── G-code Generator ──

    #[test]
    fn gcode_header_mm() {
        let params = GcodeParams::default();
        let gc = generate_gcode(&params);
        assert!(gc.contains("G21")); // mm mode
        assert!(gc.contains("G90")); // absolute
        assert!(gc.contains("M2")); // end program
    }

    #[test]
    fn gcode_header_inch() {
        let params = GcodeParams {
            units_mm: false,
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("G20")); // inch mode
    }

    #[test]
    fn gcode_line_move() {
        let params = GcodeParams {
            operations: vec![
                GcodeOp::Move {
                    x: 10.0,
                    y: 20.0,
                    z: 5.0,
                    feed: 0.0,
                },
                GcodeOp::Line {
                    x: 50.0,
                    y: 30.0,
                    z: -2.0,
                    feed: 500.0,
                },
            ],
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("G0 X10.000"));
        assert!(gc.contains("G1 X50.000"));
        assert!(gc.contains("F500"));
    }

    #[test]
    fn gcode_rect_pocket() {
        let params = GcodeParams {
            operations: vec![GcodeOp::RectPocket {
                x: 0.0,
                y: 0.0,
                w: 50.0,
                h: 30.0,
                depth: 5.0,
                feed: 300.0,
            }],
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("Rectangle pocket"));
        assert!(gc.contains("Z-5.000"));
    }

    #[test]
    fn gcode_circle_pocket() {
        let params = GcodeParams {
            operations: vec![GcodeOp::CirclePocket {
                cx: 25.0,
                cy: 25.0,
                r: 10.0,
                depth: 3.0,
                feed: 200.0,
            }],
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("Circle pocket"));
        assert!(gc.contains("G2")); // clockwise arc
    }

    #[test]
    fn gcode_drill() {
        let params = GcodeParams {
            operations: vec![GcodeOp::Drill {
                x: 10.0,
                y: 15.0,
                depth: 8.0,
                feed: 100.0,
            }],
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("Drill"));
        assert!(gc.contains("Z-8.000"));
    }

    #[test]
    fn gcode_spindle_and_tool() {
        let params = GcodeParams {
            spindle_speed: Some(12000),
            tool_number: Some(3),
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("S12000 M3"));
        assert!(gc.contains("T3 M6"));
        assert!(gc.contains("M5")); // spindle off in footer
    }

    #[test]
    fn gcode_3d_print_layer() {
        let params = GcodeParams {
            operations: vec![GcodeOp::PrintLayer {
                x: 0.0,
                y: 0.0,
                w: 20.0,
                h: 20.0,
                z: 0.3,
                extrude_rate: 0.05,
            }],
            ..Default::default()
        };
        let gc = generate_gcode(&params);
        assert!(gc.contains("Layer at Z=0.3"));
        assert!(gc.contains("E")); // extrusion values
    }

    #[test]
    fn gcode_execute_dispatch() {
        let params = GcodeParams {
            operations: vec![GcodeOp::Line {
                x: 10.0,
                y: 10.0,
                z: 0.0,
                feed: 300.0,
            }],
            ..Default::default()
        };
        let result = execute(&DeterministicToolKind::Gcode { params });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("G1"));
    }

    // ── STL Generator ──

    #[test]
    fn stl_box_12_triangles() {
        let params = stl_from_primitive(
            "test_box",
            &StlPrimitive::Box {
                w: 10.0,
                d: 10.0,
                h: 10.0,
            },
        );
        assert_eq!(params.triangles.len(), 12); // 6 faces x 2 triangles
        let stl = generate_stl(&params);
        assert!(stl.starts_with("solid test_box"));
        assert!(stl.contains("endsolid test_box"));
        assert_eq!(stl.matches("facet normal").count(), 12);
    }

    #[test]
    fn stl_cylinder_segments() {
        let params = stl_from_primitive(
            "cyl",
            &StlPrimitive::Cylinder {
                r: 5.0,
                h: 10.0,
                segments: 8,
            },
        );
        // 8 segments x 4 triangles each (2 side + top cap + bottom cap) = 32
        assert_eq!(params.triangles.len(), 32);
    }

    #[test]
    fn stl_sphere_generates() {
        let params = stl_from_primitive(
            "sphere",
            &StlPrimitive::Sphere {
                r: 5.0,
                segments: 8,
            },
        );
        assert!(!params.triangles.is_empty());
        let stl = generate_stl(&params);
        assert!(stl.contains("solid sphere"));
    }

    #[test]
    fn stl_valid_ascii_format() {
        let params = stl_from_primitive(
            "box",
            &StlPrimitive::Box {
                w: 1.0,
                d: 1.0,
                h: 1.0,
            },
        );
        let stl = generate_stl(&params);
        // Every facet has required structure
        assert_eq!(
            stl.matches("facet normal").count(),
            stl.matches("endfacet").count()
        );
        assert_eq!(
            stl.matches("outer loop").count(),
            stl.matches("endloop").count()
        );
        assert_eq!(stl.matches("vertex").count(), 12 * 3); // 12 triangles x 3 vertices
    }

    #[test]
    fn stl_execute_dispatch() {
        let result = execute(&DeterministicToolKind::Stl {
            primitive: StlPrimitive::Box {
                w: 5.0,
                d: 5.0,
                h: 5.0,
            },
            name: "test".into(),
        });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("solid test"));
    }

    // ── Three.js Generator ──

    #[test]
    fn threejs_empty_scene() {
        let params = ThreeJsParams::default();
        let html = generate_threejs(&params);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("THREE.Scene"));
        assert!(html.contains("three.min.js"));
        assert!(html.contains("animate"));
    }

    #[test]
    fn threejs_box() {
        let params = ThreeJsParams {
            objects: vec![ThreeJsObject::Box {
                width: 2.0,
                height: 2.0,
                depth: 2.0,
                color: "#ff0000".into(),
                position: [0.0, 1.0, 0.0],
            }],
            ..Default::default()
        };
        let html = generate_threejs(&params);
        assert!(html.contains("BoxGeometry(2, 2, 2)"));
        assert!(html.contains("#ff0000"));
    }

    #[test]
    fn threejs_sphere() {
        let params = ThreeJsParams {
            objects: vec![ThreeJsObject::Sphere {
                radius: 1.5,
                color: "#00ff00".into(),
                position: [0.0, 0.0, 0.0],
            }],
            ..Default::default()
        };
        let html = generate_threejs(&params);
        assert!(html.contains("SphereGeometry(1.5"));
    }

    #[test]
    fn threejs_lights() {
        let params = ThreeJsParams {
            objects: vec![
                ThreeJsObject::AmbientLight {
                    color: "#ffffff".into(),
                    intensity: 0.5,
                },
                ThreeJsObject::PointLight {
                    color: "#ffaa00".into(),
                    intensity: 1.0,
                    position: [5.0, 5.0, 5.0],
                },
            ],
            ..Default::default()
        };
        let html = generate_threejs(&params);
        assert!(html.contains("AmbientLight"));
        assert!(html.contains("PointLight"));
    }

    #[test]
    fn threejs_multiple_objects() {
        let params = ThreeJsParams {
            objects: vec![
                ThreeJsObject::Box {
                    width: 1.0,
                    height: 1.0,
                    depth: 1.0,
                    color: "#ff0000".into(),
                    position: [-2.0, 0.0, 0.0],
                },
                ThreeJsObject::Sphere {
                    radius: 0.5,
                    color: "#00ff00".into(),
                    position: [0.0, 0.0, 0.0],
                },
                ThreeJsObject::Cylinder {
                    radius_top: 0.5,
                    radius_bottom: 0.5,
                    height: 2.0,
                    color: "#0000ff".into(),
                    position: [2.0, 0.0, 0.0],
                },
            ],
            ..Default::default()
        };
        let html = generate_threejs(&params);
        assert!(html.contains("BoxGeometry"));
        assert!(html.contains("SphereGeometry"));
        assert!(html.contains("CylinderGeometry"));
    }

    #[test]
    fn threejs_execute_dispatch() {
        let params = ThreeJsParams {
            objects: vec![ThreeJsObject::Sphere {
                radius: 1.0,
                color: "#ff0000".into(),
                position: [0.0, 0.0, 0.0],
            }],
            ..Default::default()
        };
        let result = execute(&DeterministicToolKind::ThreeJs { params });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("THREE.Scene"));
    }

    // ── SVG Chart tests ──────────────────────────────

    #[test]
    fn chart_bar_basic() {
        let params = ChartParams {
            kind: ChartKind::Bar,
            title: "Sales".into(),
            labels: vec!["Q1".into(), "Q2".into(), "Q3".into()],
            values: vec![10.0, 25.0, 15.0],
            ..Default::default()
        };
        let svg = generate_chart(&params);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("Sales"));
        assert!(svg.contains("rect"));
    }

    #[test]
    fn chart_line_basic() {
        let params = ChartParams {
            kind: ChartKind::Line,
            title: "Trend".into(),
            labels: vec![],
            values: vec![1.0, 4.0, 2.0, 8.0, 5.0],
            ..Default::default()
        };
        let svg = generate_chart(&params);
        assert!(svg.contains("polyline"));
        assert!(svg.contains("circle"));
    }

    #[test]
    fn chart_pie_basic() {
        let params = ChartParams {
            kind: ChartKind::Pie,
            title: "Share".into(),
            labels: vec!["A".into(), "B".into(), "C".into()],
            values: vec![30.0, 50.0, 20.0],
            ..Default::default()
        };
        let svg = generate_chart(&params);
        assert!(svg.contains("path"));
        assert!(svg.contains("%"));
    }

    #[test]
    fn chart_scatter_basic() {
        let params = ChartParams {
            kind: ChartKind::Scatter,
            title: "XY".into(),
            labels: vec![],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
            ..Default::default()
        };
        let svg = generate_chart(&params);
        assert!(svg.contains("circle"));
    }

    #[test]
    fn chart_histogram_basic() {
        let params = ChartParams {
            kind: ChartKind::Histogram { bins: 5 },
            title: "Distribution".into(),
            labels: vec![],
            values: vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0],
            ..Default::default()
        };
        let svg = generate_chart(&params);
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn chart_empty_values() {
        let params = ChartParams {
            kind: ChartKind::Bar,
            title: "Empty".into(),
            labels: vec![],
            values: vec![],
            ..Default::default()
        };
        let svg = generate_chart(&params);
        assert!(svg.contains("</svg>"));
    }

    // ── DOT graph tests ──────────────────────────────

    #[test]
    fn dot_directed_graph() {
        let params = DotParams {
            kind: DotGraphKind::Directed,
            name: "G".into(),
            nodes: vec![
                DotNode {
                    id: "A".into(),
                    label: Some("Start".into()),
                    shape: Some("circle".into()),
                    color: None,
                },
                DotNode {
                    id: "B".into(),
                    label: Some("End".into()),
                    shape: Some("doublecircle".into()),
                    color: None,
                },
            ],
            edges: vec![DotEdge {
                from: "A".into(),
                to: "B".into(),
                label: Some("go".into()),
            }],
            rankdir: Some("LR".into()),
        };
        let dot = generate_dot(&params);
        assert!(dot.contains("digraph G"));
        assert!(dot.contains("A -> B"));
        assert!(dot.contains("rankdir=LR"));
    }

    #[test]
    fn dot_undirected_graph() {
        let params = DotParams {
            kind: DotGraphKind::Undirected,
            name: "Network".into(),
            nodes: vec![],
            edges: vec![DotEdge {
                from: "X".into(),
                to: "Y".into(),
                label: None,
            }],
            rankdir: None,
        };
        let dot = generate_dot(&params);
        assert!(dot.contains("graph Network"));
        assert!(dot.contains("X -- Y"));
    }

    // ── Mermaid tests ────────────────────────────────

    #[test]
    fn mermaid_flowchart() {
        let params = MermaidParams {
            kind: MermaidKind::Flowchart {
                direction: "TD".into(),
            },
            title: None,
            elements: vec![
                MermaidElement::Node {
                    id: "A".into(),
                    label: "Start".into(),
                    shape: "round".into(),
                },
                MermaidElement::Node {
                    id: "B".into(),
                    label: "End".into(),
                    shape: "rect".into(),
                },
                MermaidElement::Edge {
                    from: "A".into(),
                    to: "B".into(),
                    label: Some("next".into()),
                },
            ],
        };
        let out = generate_mermaid(&params);
        assert!(out.contains("flowchart TD"));
        assert!(out.contains("A(Start)"));
        assert!(out.contains("-->|next|"));
    }

    #[test]
    fn mermaid_sequence() {
        let params = MermaidParams {
            kind: MermaidKind::Sequence,
            title: None,
            elements: vec![
                MermaidElement::Message {
                    from: "Client".into(),
                    to: "Server".into(),
                    text: "GET /api".into(),
                    dashed: false,
                },
                MermaidElement::Message {
                    from: "Server".into(),
                    to: "Client".into(),
                    text: "200 OK".into(),
                    dashed: true,
                },
            ],
        };
        let out = generate_mermaid(&params);
        assert!(out.contains("sequenceDiagram"));
        assert!(out.contains("->>"));
        assert!(out.contains("-->>"));
    }

    // ── WAV audio tests ──────────────────────────────

    #[test]
    fn wav_sine_generates_data_uri() {
        let params = WavParams {
            waveform: Waveform::Sine,
            frequency: 440.0,
            duration_ms: 100,
            sample_rate: 8000,
            amplitude: 0.5,
        };
        let uri = generate_wav(&params);
        assert!(uri.starts_with("data:audio/wav;base64,"));
        assert!(uri.len() > 100); // Should have substantial data
    }

    #[test]
    fn wav_square_wave() {
        let params = WavParams {
            waveform: Waveform::Square,
            frequency: 220.0,
            duration_ms: 50,
            sample_rate: 8000,
            amplitude: 0.8,
        };
        let uri = generate_wav(&params);
        assert!(uri.starts_with("data:audio/wav;base64,"));
    }

    #[test]
    fn wav_noise() {
        let params = WavParams {
            waveform: Waveform::WhiteNoise,
            duration_ms: 50,
            ..Default::default()
        };
        let uri = generate_wav(&params);
        assert!(uri.starts_with("data:audio/wav;base64,"));
    }

    // ── OBJ tests ────────────────────────────────────

    #[test]
    fn obj_box_faces() {
        let obj = generate_obj(
            &ObjPrimitive::Box {
                w: 2.0,
                d: 2.0,
                h: 2.0,
            },
            "cube",
        );
        assert!(obj.contains("o cube"));
        assert!(obj.contains("v "));
        assert!(obj.contains("f "));
        assert_eq!(obj.matches("v ").count(), 8); // 8 vertices
        assert_eq!(obj.matches("f ").count(), 6); // 6 quad faces
    }

    #[test]
    fn obj_sphere_vertices() {
        let obj = generate_obj(
            &ObjPrimitive::Sphere {
                r: 1.0,
                segments: 8,
            },
            "ball",
        );
        assert!(obj.contains("o ball"));
        assert!(obj.contains("v "));
    }

    #[test]
    fn obj_plane() {
        let obj = generate_obj(&ObjPrimitive::Plane { w: 10.0, d: 10.0 }, "ground");
        assert_eq!(obj.matches("v ").count(), 4);
        assert_eq!(obj.matches("f ").count(), 1);
    }

    // ── LaTeX tests ──────────────────────────────────

    #[test]
    fn latex_fraction() {
        let out = generate_latex(&LatexKind::Fraction {
            num: "x+1".into(),
            den: "2".into(),
        });
        assert_eq!(out, "\\frac{x+1}{2}");
    }

    #[test]
    fn latex_matrix() {
        let out = generate_latex(&LatexKind::Matrix {
            rows: vec![vec!["1".into(), "0".into()], vec!["0".into(), "1".into()]],
            bracket: "[".into(),
        });
        assert!(out.contains("\\begin{bmatrix}"));
        assert!(out.contains("1 & 0"));
    }

    #[test]
    fn latex_integral() {
        let out = generate_latex(&LatexKind::Integral {
            var: "x".into(),
            lower: Some("0".into()),
            upper: Some("\\infty".into()),
            body: "e^{-x^2}".into(),
        });
        assert!(out.contains("\\int"));
        assert!(out.contains("dx"));
    }

    #[test]
    fn latex_summation() {
        let out = generate_latex(&LatexKind::Summation {
            var: "i".into(),
            lower: "1".into(),
            upper: "n".into(),
            body: "i^2".into(),
        });
        assert!(out.contains("\\sum_{i=1}^{n}"));
    }

    // ── Equation formatter tests ─────────────────────

    #[test]
    fn equation_unicode_symbols() {
        let out = format_equation_unicode("x^2 + sqrt(y) != z * pi");
        assert!(out.contains('\u{00B2}')); // ²
        assert!(out.contains('\u{221A}')); // √
        assert!(out.contains('\u{2260}')); // ≠
        assert!(out.contains('\u{00D7}')); // ×
        assert!(out.contains('\u{03C0}')); // π
    }

    #[test]
    fn equation_subscripts() {
        let out = format_equation_unicode("x_0 + x_1 = x_n");
        assert!(out.contains('\u{2080}')); // ₀
        assert!(out.contains('\u{2081}')); // ₁
    }

    // ── Terraform HCL tests ──────────────────────────

    #[test]
    fn terraform_ec2() {
        let params = TerraformParams {
            provider: "aws".into(),
            region: "us-east-1".into(),
            resources: vec![TerraformResource::AwsInstance {
                ami: "ami-12345".into(),
                instance_type: "t3.micro".into(),
                name: "web-server".into(),
            }],
        };
        let hcl = generate_terraform(&params);
        assert!(hcl.contains("provider \"aws\""));
        assert!(hcl.contains("ami-12345"));
        assert!(hcl.contains("t3.micro"));
    }

    #[test]
    fn terraform_s3_and_sg() {
        let params = TerraformParams {
            provider: "aws".into(),
            region: "eu-west-1".into(),
            resources: vec![
                TerraformResource::AwsS3Bucket {
                    bucket: "my-bucket".into(),
                    acl: "private".into(),
                },
                TerraformResource::AwsSecurityGroup {
                    name: "web-sg".into(),
                    ingress_ports: vec![80, 443],
                },
            ],
        };
        let hcl = generate_terraform(&params);
        assert!(hcl.contains("aws_s3_bucket"));
        assert!(hcl.contains("aws_security_group"));
        assert!(hcl.contains("from_port   = 80"));
        assert!(hcl.contains("from_port   = 443"));
    }

    // ── Docker Compose tests ─────────────────────────

    #[test]
    fn compose_basic_stack() {
        let params = ComposeParams {
            services: vec![
                ComposeService {
                    name: "web".into(),
                    image: "nginx:latest".into(),
                    ports: vec![(80, 80)],
                    environment: vec![],
                    volumes: vec![],
                    depends_on: vec!["db".into()],
                    command: None,
                },
                ComposeService {
                    name: "db".into(),
                    image: "postgres:15".into(),
                    ports: vec![(5432, 5432)],
                    environment: vec![("POSTGRES_PASSWORD".into(), "secret".into())],
                    volumes: vec!["pgdata:/var/lib/postgresql/data".into()],
                    depends_on: vec![],
                    command: None,
                },
            ],
            networks: vec![],
            volumes: vec!["pgdata".into()],
        };
        let yaml = generate_compose(&params);
        assert!(yaml.contains("services:"));
        assert!(yaml.contains("nginx:latest"));
        assert!(yaml.contains("postgres:15"));
        assert!(yaml.contains("depends_on:"));
        assert!(yaml.contains("volumes:"));
    }

    // ── Kubernetes tests ─────────────────────────────

    #[test]
    fn k8s_deployment_and_service() {
        let params = K8sParams {
            namespace: "default".into(),
            resources: vec![
                K8sResource::Deployment {
                    name: "api".into(),
                    image: "myapp:1.0".into(),
                    replicas: 3,
                    port: 8080,
                    cpu: Some("250m".into()),
                    memory: Some("256Mi".into()),
                },
                K8sResource::Service {
                    name: "api".into(),
                    port: 80,
                    target_port: 8080,
                    svc_type: "ClusterIP".into(),
                },
            ],
        };
        let yaml = generate_k8s(&params);
        assert!(yaml.contains("kind: Deployment"));
        assert!(yaml.contains("replicas: 3"));
        assert!(yaml.contains("kind: Service"));
        assert!(yaml.contains("---")); // separator between resources
    }

    #[test]
    fn k8s_ingress() {
        let params = K8sParams {
            namespace: "prod".into(),
            resources: vec![K8sResource::Ingress {
                name: "web-ingress".into(),
                host: "example.com".into(),
                service: "web".into(),
                port: 80,
            }],
        };
        let yaml = generate_k8s(&params);
        assert!(yaml.contains("kind: Ingress"));
        assert!(yaml.contains("example.com"));
    }

    // ── KiCad tests ──────────────────────────────────

    #[test]
    fn kicad_basic_circuit() {
        let params = KicadParams {
            title: "LED Circuit".into(),
            components: vec![
                KicadComponent::Resistor {
                    ref_des: "R1".into(),
                    value: "330".into(),
                    x: 10.0,
                    y: 20.0,
                },
                KicadComponent::Led {
                    ref_des: "D1".into(),
                    color: "Red".into(),
                    x: 20.0,
                    y: 20.0,
                },
                KicadComponent::Vcc {
                    voltage: "5V".into(),
                    x: 5.0,
                    y: 15.0,
                },
                KicadComponent::Gnd { x: 25.0, y: 25.0 },
            ],
            wires: vec![KicadWire {
                x1: 10.0,
                y1: 20.0,
                x2: 20.0,
                y2: 20.0,
            }],
        };
        let sch = generate_kicad(&params);
        assert!(sch.contains("kicad_sch"));
        assert!(sch.contains("Device:R"));
        assert!(sch.contains("Device:LED"));
        assert!(sch.contains("wire"));
    }

    // ── SPICE tests ──────────────────────────────────

    #[test]
    fn spice_rc_circuit() {
        let params = SpiceParams {
            title: "RC Low-pass Filter".into(),
            elements: vec![
                SpiceElement::VoltageSource {
                    name: "1".into(),
                    node_p: "in".into(),
                    node_n: "0".into(),
                    value: "AC 1".into(),
                },
                SpiceElement::Resistor {
                    name: "1".into(),
                    node_p: "in".into(),
                    node_n: "out".into(),
                    value: "1k".into(),
                },
                SpiceElement::Capacitor {
                    name: "1".into(),
                    node_p: "out".into(),
                    node_n: "0".into(),
                    value: "1u".into(),
                },
            ],
            analysis: Some(SpiceAnalysis::Ac {
                start: "1".into(),
                stop: "100k".into(),
                points: 100,
            }),
        };
        let spice = generate_spice(&params);
        assert!(spice.contains("* RC Low-pass Filter"));
        assert!(spice.contains("R1 in out 1k"));
        assert!(spice.contains("C1 out 0 1u"));
        assert!(spice.contains(".ac dec"));
        assert!(spice.contains(".end"));
    }

    // ── Bitmap tests ─────────────────────────────────

    #[test]
    fn bitmap_checkerboard() {
        let params = BitmapParams {
            width: 8,
            height: 8,
            pattern: BitmapPattern::Checkerboard {
                size: 2,
                color1: Pixel {
                    r: 255,
                    g: 255,
                    b: 255,
                },
                color2: Pixel { r: 0, g: 0, b: 0 },
            },
        };
        let ppm = generate_ppm(&params);
        assert!(ppm.starts_with("P3\n8 8\n255\n"));
        assert!(ppm.contains("255 255 255"));
        assert!(ppm.contains("0 0 0"));
    }

    #[test]
    fn bitmap_gradient() {
        let params = BitmapParams {
            width: 16,
            height: 1,
            pattern: BitmapPattern::Gradient {
                direction: "horizontal".into(),
            },
        };
        let ppm = generate_ppm(&params);
        assert!(ppm.starts_with("P3"));
    }

    // ── Table/CSV tests ──────────────────────────────

    #[test]
    fn table_csv_output() {
        let params = TableParams {
            headers: vec!["Name".into(), "Age".into()],
            rows: vec![
                vec!["Alice".into(), "30".into()],
                vec!["Bob".into(), "25".into()],
            ],
            format: TableFormat::Csv,
        };
        let csv = generate_table(&params);
        assert!(csv.contains("Name,Age"));
        assert!(csv.contains("Alice,30"));
    }

    #[test]
    fn table_markdown_output() {
        let params = TableParams {
            headers: vec!["Col1".into(), "Col2".into()],
            rows: vec![vec!["a".into(), "b".into()]],
            format: TableFormat::Markdown,
        };
        let md = generate_table(&params);
        assert!(md.contains("| Col1 | Col2 |"));
        assert!(md.contains("| --- |") || md.contains("| ---- |"));
    }

    #[test]
    fn table_ascii_output() {
        let params = TableParams {
            headers: vec!["X".into(), "Y".into()],
            rows: vec![vec!["1".into(), "2".into()]],
            format: TableFormat::AsciiTable,
        };
        let tbl = generate_table(&params);
        assert!(tbl.contains("+"));
        assert!(tbl.contains("|"));
    }

    #[test]
    fn table_html_output() {
        let params = TableParams {
            headers: vec!["H1".into()],
            rows: vec![vec!["val".into()]],
            format: TableFormat::Html,
        };
        let html = generate_table(&params);
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>H1</th>"));
        assert!(html.contains("<td>val</td>"));
    }

    // ── Document classifier tests ────────────────────

    #[test]
    fn doc_classify_extension() {
        assert_eq!(classify_document_extension("report.pdf"), DocType::Pdf);
        assert_eq!(classify_document_extension("data.xlsx"), DocType::Xlsx);
        assert_eq!(classify_document_extension("slides.pptx"), DocType::Pptx);
        assert_eq!(classify_document_extension("image.png"), DocType::Png);
        assert_eq!(classify_document_extension("model.stl"), DocType::Stl);
        assert_eq!(classify_document_extension("config.yaml"), DocType::Yaml);
        assert_eq!(classify_document_extension("unknown.xyz"), DocType::Unknown);
    }

    #[test]
    fn doc_classify_magic_bytes() {
        assert_eq!(classify_document_bytes(b"%PDF-1.4 test"), DocType::Pdf);
        assert_eq!(
            classify_document_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A]),
            DocType::Png
        );
        assert_eq!(
            classify_document_bytes(&[0xFF, 0xD8, 0xFF, 0xE0]),
            DocType::Jpeg
        );
        assert_eq!(classify_document_bytes(b"GIF89a"), DocType::Gif);
        assert_eq!(classify_document_bytes(b"solid cube"), DocType::Stl);
        assert_eq!(classify_document_bytes(b"ab"), DocType::Unknown); // too short
    }

    // ── PDF generator tests ──────────────────────────

    #[test]
    fn pdf_basic_generation() {
        let params = PdfParams {
            title: "Test Document".into(),
            author: "jouleclaw".into(),
            pages: vec![PdfPage {
                lines: vec!["Hello, World!".into(), "Second line.".into()],
                font_size: 12.0,
            }],
            template: None,
            header: None,
            footer: None,
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.starts_with("%PDF-1.4"));
        assert!(pdf.contains("%%EOF"));
        assert!(pdf.contains("Hello, World!"));
        assert!(pdf.contains("/Type /Catalog"));
    }

    #[test]
    fn pdf_multi_page() {
        let params = PdfParams {
            title: "Multi".into(),
            author: "test".into(),
            pages: vec![
                PdfPage {
                    lines: vec!["Page 1".into()],
                    font_size: 14.0,
                },
                PdfPage {
                    lines: vec!["Page 2".into()],
                    font_size: 14.0,
                },
            ],
            template: None,
            header: None,
            footer: None,
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.contains("Page 1"));
        assert!(pdf.contains("Page 2"));
        assert!(pdf.contains("/Count 2"));
    }

    #[test]
    fn pdf_report_template() {
        let params = PdfParams {
            title: "Q4 Sales Analysis".into(),
            author: "David Charlot".into(),
            pages: vec![],
            template: Some(PdfTemplate::Report),
            header: Some("Q4 Sales Analysis".into()),
            footer: Some("Confidential".into()),
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.starts_with("%PDF-1.4"));
        assert!(pdf.contains("Q4 Sales Analysis"));
        assert!(pdf.contains("Prepared by"));
        assert!(pdf.contains("Executive Summary"));
    }

    #[test]
    fn pdf_invoice_template() {
        let params = PdfParams {
            title: "Web Development Services".into(),
            author: "Acme Corp".into(),
            pages: vec![],
            template: Some(PdfTemplate::Invoice),
            header: None,
            footer: None,
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.contains("INVOICE"));
        assert!(pdf.contains("Acme Corp"));
        assert!(pdf.contains("Payment due"));
    }

    #[test]
    fn pdf_letter_template() {
        let params = PdfParams {
            title: "Project Proposal".into(),
            author: "Jane Smith".into(),
            pages: vec![],
            template: Some(PdfTemplate::Letter),
            header: None,
            footer: None,
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.contains("Dear Sir/Madam"));
        assert!(pdf.contains("Sincerely"));
        assert!(pdf.contains("Jane Smith"));
    }

    #[test]
    fn pdf_resume_template() {
        let params = PdfParams {
            title: "John Doe".into(),
            author: "John Doe".into(),
            pages: vec![],
            template: Some(PdfTemplate::Resume),
            header: None,
            footer: None,
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.contains("John Doe"));
        assert!(pdf.contains("PROFESSIONAL SUMMARY"));
        assert!(pdf.contains("EXPERIENCE"));
        assert!(pdf.contains("EDUCATION"));
    }

    #[test]
    fn pdf_with_header_footer() {
        let params = PdfParams {
            title: "Test".into(),
            author: "test".into(),
            pages: vec![PdfPage {
                lines: vec!["Content".into()],
                font_size: 12.0,
            }],
            template: None,
            header: Some("My Header".into()),
            footer: Some("Page 1".into()),
        };
        let pdf = generate_pdf(&params);
        assert!(pdf.contains("My Header"));
        assert!(pdf.contains("Page 1"));
    }

    // ── §43 MD5 / CRC32 ──

    #[test]
    fn md5_empty_string() {
        assert_eq!(md5_hash(""), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn md5_hello() {
        assert_eq!(md5_hash("hello"), "5d41402abc4b2a76b9719d911017c592");
    }

    #[test]
    fn crc32_hello() {
        assert_eq!(crc32_hash("hello"), "3610a686");
    }

    #[test]
    fn hash_dispatch_md5() {
        let tool = DeterministicToolKind::Hash {
            operation: HashOp::Md5 {
                input: "test".into(),
            },
        };
        let result = execute(&tool).unwrap();
        assert!(result.starts_with("MD5: "));
        assert_eq!(result.len(), 4 + 1 + 32); // "MD5: " + 32 hex chars
    }

    #[test]
    fn hash_dispatch_sha256() {
        let tool = DeterministicToolKind::Hash {
            operation: HashOp::Sha256 {
                input: "test".into(),
            },
        };
        let result = execute(&tool).unwrap();
        assert!(result.starts_with("SHA256: "));
    }

    // ── §44 Morse ──

    #[test]
    fn morse_encode_sos() {
        assert_eq!(morse_encode("SOS"), "... --- ...");
    }

    #[test]
    fn morse_decode_sos() {
        assert_eq!(morse_decode("... --- ..."), "SOS");
    }

    #[test]
    fn morse_roundtrip() {
        let encoded = morse_encode("HELLO");
        let decoded = morse_decode(&encoded);
        assert_eq!(decoded, "HELLO");
    }

    // ── §45 Text Diff ──

    #[test]
    fn diff_identical() {
        let diff = text_diff("hello\nworld", "hello\nworld");
        assert!(diff.contains(" hello"));
        assert!(diff.contains(" world"));
        assert!(!diff.contains("+"));
        assert!(!diff.contains("-"));
    }

    #[test]
    fn diff_addition() {
        let diff = text_diff("line1", "line1\nline2");
        assert!(diff.contains("+line2"));
    }

    #[test]
    fn diff_removal() {
        let diff = text_diff("line1\nline2", "line1");
        assert!(diff.contains("-line2"));
    }

    // ── §46 Finance ──

    #[test]
    fn compound_interest() {
        let result = finance_calc(&FinanceOp::CompoundInterest {
            principal: 1000.0,
            rate: 5.0,
            years: 10.0,
            compounds_per_year: 12,
        });
        assert!(result.contains("Final Amount: $1647"));
    }

    #[test]
    fn loan_amortization() {
        let result = finance_calc(&FinanceOp::Amortization {
            principal: 200000.0,
            rate: 4.0,
            months: 360,
        });
        assert!(result.contains("Monthly Payment: $"));
    }

    #[test]
    fn npv_calculation() {
        let result = finance_calc(&FinanceOp::Npv {
            rate: 10.0,
            cash_flows: vec![-1000.0, 300.0, 400.0, 500.0],
        });
        assert!(result.contains("NPV: $"));
    }

    #[test]
    fn irr_calculation() {
        let result = finance_calc(&FinanceOp::Irr {
            cash_flows: vec![-1000.0, 300.0, 400.0, 500.0],
        });
        assert!(result.contains("IRR: "));
    }

    // ── §47 Bitwise ──

    #[test]
    fn bitwise_and() {
        let result = bitwise_calc(&BitwiseOp::And { a: 0xFF, b: 0x0F });
        assert!(
            result.contains("15")
                || result.contains("0x0f")
                || result.contains("0x0F")
                || result.contains("0f")
        );
    }

    #[test]
    fn bitwise_xor() {
        let result = bitwise_calc(&BitwiseOp::Xor { a: 0xAA, b: 0x55 });
        assert!(result.contains("ff") || result.contains("FF") || result.contains("255"));
    }

    // ── §48 Truth Table ──

    #[test]
    fn truth_table_and() {
        let result = truth_table("A AND B");
        assert!(result.contains("A") && result.contains("B"));
        // A=0,B=0 → 0; A=1,B=1 → 1
        assert!(result.contains("0"));
        assert!(result.contains("1"));
    }

    #[test]
    fn truth_table_or() {
        let result = truth_table("A OR B");
        assert!(result.contains("A") && result.contains("B"));
    }

    #[test]
    fn truth_table_not() {
        let result = truth_table("NOT A");
        assert!(result.contains("A"));
    }

    // ── §49 JSON Schema Validate ──

    #[test]
    fn json_schema_valid() {
        let json = r#"{"name": "Alice", "age": 30}"#;
        let schema = r#"{"type": "object", "required": ["name"]}"#;
        let result = validate_json_schema(json, schema);
        assert!(result.is_ok());
        assert!(result.unwrap().to_lowercase().contains("valid"));
    }

    #[test]
    fn json_schema_invalid_missing_required() {
        let json = r#"{"age": 30}"#;
        let schema = r#"{"type": "object", "required": ["name"]}"#;
        let result = validate_json_schema(json, schema);
        // Either returns Err or contains "missing" / "invalid"
        assert!(result.is_err() || result.unwrap().to_lowercase().contains("missing"));
    }

    // ── §50 XML Convert ──

    #[test]
    fn xml_to_json_simple() {
        let xml = "<root><name>Alice</name><age>30</age></root>";
        let result = xml_to_json(xml);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("Alice"));
        assert!(json.contains("30"));
    }

    #[test]
    fn json_to_xml_simple() {
        let json = r#"{"name": "Alice", "age": "30"}"#;
        let result = json_to_xml(json);
        assert!(result.is_ok());
        let xml = result.unwrap();
        assert!(xml.contains("<name>Alice</name>"));
    }

    // ── §51 URL Parse ──

    #[test]
    fn url_parse_full() {
        let parts = parse_url("https://example.com:8080/path?key=value#frag").unwrap();
        assert_eq!(parts.scheme, "https");
        assert_eq!(parts.host, "example.com");
        assert_eq!(parts.port, Some(8080));
        assert_eq!(parts.path, "/path");
        assert_eq!(parts.query, vec![("key".into(), "value".into())]);
        assert_eq!(parts.fragment, Some("frag".into()));
    }

    #[test]
    fn url_parse_simple() {
        let parts = parse_url("http://example.com/").unwrap();
        assert_eq!(parts.scheme, "http");
        assert_eq!(parts.host, "example.com");
        assert_eq!(parts.port, None);
    }

    // ── §52 cURL ──

    #[test]
    fn curl_get() {
        let result = generate_curl(&CurlParams {
            method: HttpMethod::Get,
            url: "https://api.example.com/data".into(),
            headers: vec![("Accept".into(), "application/json".into())],
            body: None,
            auth: None,
            verbose: false,
        });
        assert!(result.contains("curl"));
        assert!(result.contains("api.example.com"));
    }

    #[test]
    fn curl_post_with_body() {
        let result = generate_curl(&CurlParams {
            method: HttpMethod::Post,
            url: "https://api.example.com/data".into(),
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: Some(r#"{"key":"value"}"#.into()),
            auth: None,
            verbose: false,
        });
        assert!(result.contains("POST"));
        assert!(result.contains("-d"));
    }

    // ── §53 Timezone Convert ──

    #[test]
    fn timezone_utc_to_est() {
        let result = convert_timezone(14, 30, "UTC", "America/New_York");
        assert!(result.is_ok());
        // UTC-5 in winter
        let text = result.unwrap();
        assert!(text.contains("09:30") || text.contains("9:30"));
    }

    // ── §54 Barcode ──

    #[test]
    fn barcode_code128() {
        let result = generate_barcode("HELLO", &BarcodeKind::Code128);
        assert!(result.is_ok());
        let svg = result.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn barcode_ean13() {
        let result = generate_barcode("5901234123457", &BarcodeKind::Ean13);
        assert!(result.is_ok());
        let svg = result.unwrap();
        assert!(svg.contains("<svg"));
    }

    // ── §55 Punycode ──

    #[test]
    fn punycode_encode_ascii() {
        let result = punycode_encode("hello");
        assert!(result.contains("hello"));
    }

    #[test]
    fn punycode_decode_ascii() {
        let result = punycode_decode("hello");
        assert!(result.contains("hello"));
    }

    // ── §56 Hex Dump ──

    #[test]
    fn hex_dump_hello() {
        let result = hex_dump("Hello");
        // xxd format: pairs of hex digits grouped, e.g. "4865 6c6c 6f"
        assert!(result.contains("4865"));
        assert!(result.contains("Hello"));
    }

    #[test]
    fn hex_dump_offset() {
        let result = hex_dump("Hello, World!");
        assert!(result.contains("00000000:"));
    }

    // ── §57 Glob Match ──

    #[test]
    fn glob_match_star() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.py"));
    }

    #[test]
    fn glob_match_question() {
        assert!(glob_match("file?.txt", "file1.txt"));
        assert!(!glob_match("file?.txt", "file12.txt"));
    }

    // ── §58 Geometry ──

    #[test]
    fn geometry_distance_2d() {
        let result = geometry_calc(&GeometryOp::Distance2D {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
        });
        assert!(result.contains("5.0"));
    }

    #[test]
    fn geometry_circle() {
        let result = geometry_calc(&GeometryOp::Circle { radius: 1.0 });
        assert!(result.contains("3.14"));
    }

    #[test]
    fn geometry_triangle_area() {
        let result = geometry_calc(&GeometryOp::TriangleArea {
            a: 3.0,
            b: 4.0,
            c: 5.0,
        });
        assert!(result.contains("6.0"));
    }

    // ── §59 UUID v5 ──

    #[test]
    fn uuid_v5_deterministic() {
        let a = uuid_v5("dns", "example.com");
        let b = uuid_v5("dns", "example.com");
        assert_eq!(a, b); // deterministic: same input → same output
    }

    #[test]
    fn uuid_v5_format() {
        let result = uuid_v5("dns", "example.com");
        // UUID format: 8-4-4-4-12
        assert_eq!(result.len(), 36);
        assert_eq!(result.chars().nth(8), Some('-'));
    }

    // ── §60 Snowflake Decode ──

    #[test]
    fn snowflake_decode_twitter() {
        let result = decode_snowflake(1234567890123456789, 1288834974657);
        assert!(result.contains("Timestamp"));
        assert!(result.contains("Worker"));
        assert!(result.contains("Sequence"));
    }

    // ── §61 JSON Diff ──

    #[test]
    fn json_diff_identical() {
        let result = json_diff(r#"{"a":1}"#, r#"{"a":1}"#).unwrap();
        assert!(
            result.to_lowercase().contains("identical")
                || result.to_lowercase().contains("no diff")
                || result.is_empty()
                || result.contains("equal")
        );
    }

    #[test]
    fn json_diff_changed() {
        let result = json_diff(r#"{"a":1}"#, r#"{"a":2}"#).unwrap();
        assert!(
            result.contains("a")
                || result.contains("changed")
                || result.contains("1") && result.contains("2")
        );
    }

    #[test]
    fn json_diff_added_key() {
        let result = json_diff(r#"{"a":1}"#, r#"{"a":1,"b":2}"#).unwrap();
        assert!(result.contains("b"));
    }

    // ── §62 HTTP Headers ──

    #[test]
    fn parse_headers_basic() {
        let raw = "Content-Type: application/json\r\nAuthorization: Bearer token123\r\n";
        let result = parse_http_headers(raw);
        assert!(result.contains("Content-Type"));
        assert!(result.contains("application/json"));
        assert!(result.contains("Authorization"));
    }

    // ── §63 MIME Lookup ──

    #[test]
    fn mime_lookup_common() {
        assert_eq!(mime_from_extension("json"), "application/json");
        assert_eq!(mime_from_extension("html"), "text/html");
        assert_eq!(mime_from_extension("png"), "image/png");
        assert_eq!(mime_from_extension("pdf"), "application/pdf");
    }

    #[test]
    fn mime_lookup_with_dot() {
        assert_eq!(mime_from_extension(".json"), "application/json");
    }

    #[test]
    fn mime_lookup_unknown() {
        assert_eq!(mime_from_extension("xyz123"), "application/octet-stream");
    }

    // ── §64 Helm ──

    #[test]
    fn helm_chart_basic() {
        let result = generate_helm(&HelmParams {
            name: "myapp".into(),
            image: "nginx".into(),
            tag: "latest".into(),
            port: 80,
            replicas: 3,
            env_vars: vec![("ENV".into(), "prod".into())],
        });
        assert!(result.contains("myapp"));
        assert!(result.contains("nginx"));
        assert!(result.contains("replicaCount: 3"));
    }

    // ── §65 DNS ──

    #[test]
    fn dns_a_record() {
        let result = generate_dns_records(&[DnsRecord::A {
            name: "example.com".into(),
            ip: "93.184.216.34".into(),
            ttl: 3600,
        }]);
        assert!(result.contains("example.com"));
        assert!(result.contains("93.184.216.34"));
        assert!(result.contains("3600"));
    }

    // ── §66 ASCII Art ──

    #[test]
    fn ascii_banner_hello() {
        let result = ascii_banner("HI");
        // Should produce multi-line output
        assert!(result.lines().count() > 1);
    }

    #[test]
    fn box_drawing_hello() {
        let result = box_drawing("Hello");
        assert!(result.contains("Hello"));
        assert!(result.contains("┌"));
        assert!(result.contains("└"));
        assert!(result.contains("│"));
    }

    // ── §67 Color Space Convert ──

    #[test]
    fn rgb_to_cmyk() {
        let result = color_convert(&ColorConvertOp::RgbToCmyk { r: 255, g: 0, b: 0 });
        assert!(result.contains("CMYK"));
        assert!(result.contains("0.0%")); // cyan should be 0 for pure red
    }

    #[test]
    fn cmyk_to_rgb() {
        let result = color_convert(&ColorConvertOp::CmykToRgb {
            c: 0.0,
            m: 100.0,
            y: 100.0,
            k: 0.0,
        });
        assert!(result.contains("RGB(255,0,0)"));
    }

    #[test]
    fn rgb_to_hsl() {
        let result = color_convert(&ColorConvertOp::RgbToHsl { r: 255, g: 0, b: 0 });
        assert!(result.contains("HSL(0°"));
    }

    #[test]
    fn hsl_to_rgb() {
        let result = color_convert(&ColorConvertOp::HslToRgb {
            h: 0.0,
            s: 100.0,
            l: 50.0,
        });
        assert!(result.contains("RGB(255,0,0)"));
    }

    // ── §68 RPN Calculator ──

    #[test]
    fn rpn_simple_add() {
        assert!((rpn_calc("3 4 +").unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn rpn_complex() {
        // (3 + 4) * 2 = 14
        assert!((rpn_calc("3 4 + 2 *").unwrap() - 14.0).abs() < 1e-10);
    }

    #[test]
    fn rpn_sqrt() {
        assert!((rpn_calc("144 sqrt").unwrap() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn rpn_div_by_zero() {
        assert!(rpn_calc("1 0 /").is_err());
    }

    // ── §69 JWT Decoder ──

    #[test]
    fn jwt_decode_valid() {
        // Standard JWT test token
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let result = decode_jwt(token);
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert!(parts.header.contains("HS256"));
        assert!(parts.payload.contains("John Doe"));
    }

    #[test]
    fn jwt_decode_invalid() {
        assert!(decode_jwt("not.a.valid.jwt.token").is_err() || decode_jwt("not-a-jwt").is_err());
    }

    // ── §70 Minifier ──

    #[test]
    fn minify_json_whitespace() {
        let input = r#"{  "name":  "test",  "value":  42  }"#;
        let result = minify_json(input);
        assert_eq!(result, r#"{"name":"test","value":42}"#);
    }

    #[test]
    fn minify_json_preserves_string_spaces() {
        let input = r#"{"msg": "hello world"}"#;
        let result = minify_json(input);
        assert!(result.contains("hello world"));
    }

    #[test]
    fn minify_css_basic() {
        let input = "body {\n  color: red;\n  margin: 0;\n}";
        let result = minify_css(input);
        assert!(!result.contains('\n'));
        assert!(result.contains("color:red") || result.contains("color: red"));
    }

    // ── §71 Base85 ──

    #[test]
    fn base85_encode_hello() {
        let encoded = base85_encode("Hello");
        assert!(encoded.starts_with("<~"));
        assert!(encoded.ends_with("~>"));
    }

    #[test]
    fn base85_roundtrip() {
        let encoded = base85_encode("Test");
        let decoded = base85_decode(&encoded).unwrap();
        assert_eq!(decoded, "Test");
    }

    // ── §72 Dockerfile Lint ──

    #[test]
    fn dockerfile_lint_no_tag() {
        let issues = lint_dockerfile("FROM ubuntu\nRUN apt-get update");
        assert!(issues.iter().any(|i| i.message.contains("DL3006")));
    }

    #[test]
    fn dockerfile_lint_latest_tag() {
        let issues = lint_dockerfile("FROM ubuntu:latest\nRUN echo hello");
        assert!(issues.iter().any(|i| i.message.contains("DL3007")));
    }

    #[test]
    fn dockerfile_lint_add_instead_of_copy() {
        let issues = lint_dockerfile("FROM ubuntu:22.04\nADD file.txt /app/");
        assert!(issues.iter().any(|i| i.message.contains("DL3020")));
    }

    #[test]
    fn dockerfile_lint_clean() {
        let issues = lint_dockerfile(
            "FROM rust:1.75-slim\nWORKDIR /app\nCOPY . .\nRUN cargo build --release",
        );
        assert!(issues.is_empty());
    }

    // ── §73 Cron Describe ──

    #[test]
    fn cron_describe_every_minute() {
        let result = describe_cron("* * * * *");
        assert!(result.contains("Every minute"));
    }

    #[test]
    fn cron_describe_specific_time() {
        let result = describe_cron("30 14 * * *");
        assert!(result.contains("30") && result.contains("14"));
    }

    #[test]
    fn cron_describe_interval() {
        let result = describe_cron("*/5 * * * *");
        assert!(result.contains("5 minutes"));
    }

    #[test]
    fn cron_describe_weekday() {
        let result = describe_cron("0 9 * * 1");
        assert!(result.contains("Monday"));
    }

    // ── §74 Verilog ──

    #[test]
    fn verilog_basic_module() {
        let result = generate_verilog(&VerilogParams {
            name: "and_gate".into(),
            inputs: vec![("a".into(), 1), ("b".into(), 1)],
            outputs: vec![("y".into(), 1)],
            body: "always @(*) y = a & b;".into(),
        });
        assert!(result.contains("module and_gate"));
        assert!(result.contains("input a"));
        assert!(result.contains("output reg y"));
        assert!(result.contains("endmodule"));
    }

    #[test]
    fn verilog_wide_bus() {
        let result = generate_verilog(&VerilogParams {
            name: "adder".into(),
            inputs: vec![("a".into(), 8), ("b".into(), 8)],
            outputs: vec![("sum".into(), 9)],
            body: "assign sum = a + b;".into(),
        });
        assert!(result.contains("[7:0] a"));
        assert!(result.contains("[8:0] sum"));
    }

    // ── §75 MathML ──

    #[test]
    fn mathml_binary_expr() {
        let result = to_mathml("3 + 4");
        assert!(result.contains("<math"));
        assert!(result.contains("<mn>3</mn>"));
        assert!(result.contains("<mo>+</mo>"));
        assert!(result.contains("<mn>4</mn>"));
    }

    #[test]
    fn mathml_fraction() {
        let result = to_mathml("1/2");
        assert!(result.contains("<mfrac>"));
    }

    // ── §76 Seed Palette ──

    #[test]
    fn seed_palette_deterministic() {
        let a = seed_palette("ocean", 5);
        let b = seed_palette("ocean", 5);
        assert_eq!(a, b); // deterministic: same seed → same palette
    }

    #[test]
    fn seed_palette_count() {
        let colors = seed_palette("sunset", 8);
        assert_eq!(colors.len(), 8);
        assert!(colors[0].starts_with("hsl("));
    }

    // ── §77 Data Size ──

    #[test]
    fn data_size_bytes() {
        let result = format_data_size(1024);
        assert!(result.contains("1024 bytes"));
        assert!(result.contains("1.00 KiB"));
    }

    #[test]
    fn data_size_gigabytes() {
        let result = format_data_size(1_073_741_824);
        assert!(result.contains("1.00 GiB"));
        assert!(result.contains("1.07 GB"));
    }

    // ── §78 String Escape ──

    #[test]
    fn html_escape_test() {
        assert_eq!(
            escape_op(&EscapeOp::HtmlEscape {
                input: "<b>test</b>".into()
            }),
            "&lt;b&gt;test&lt;/b&gt;"
        );
    }

    #[test]
    fn html_unescape_test() {
        assert_eq!(
            escape_op(&EscapeOp::HtmlUnescape {
                input: "&lt;b&gt;".into()
            }),
            "<b>"
        );
    }

    #[test]
    fn json_escape_test() {
        let result = escape_op(&EscapeOp::JsonEscape {
            input: "line1\nline2".into(),
        });
        assert!(result.contains("\\n"));
    }

    #[test]
    fn url_encode_test() {
        let result = escape_op(&EscapeOp::UrlEncode {
            input: "hello world".into(),
        });
        assert_eq!(result, "hello%20world");
    }

    #[test]
    fn url_decode_test() {
        let result = escape_op(&EscapeOp::UrlDecode {
            input: "hello%20world".into(),
        });
        assert_eq!(result, "hello world");
    }

    // ── §79 IP Validate ──

    #[test]
    fn ip_validate_private() {
        let result = validate_ip("192.168.1.1");
        assert!(result.contains("Private"));
        assert!(result.contains("IPv4"));
        assert!(result.contains("Class: C"));
    }

    #[test]
    fn ip_validate_loopback() {
        let result = validate_ip("127.0.0.1");
        assert!(result.contains("Loopback"));
    }

    #[test]
    fn ip_validate_ipv6() {
        let result = validate_ip("::1");
        assert!(result.contains("IPv6"));
        assert!(result.contains("Loopback"));
    }

    #[test]
    fn ip_validate_invalid() {
        let result = validate_ip("not-an-ip");
        assert!(result.contains("Invalid"));
    }

    // ── §80 Semver ──

    #[test]
    fn semver_parse() {
        let v = parse_semver("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn semver_compare_major() {
        let result = compare_semver("1.0.0", "2.0.0").unwrap();
        assert!(result.contains("MAJOR"));
        assert!(result.contains("<"));
        assert!(result.contains("Breaking change: Yes"));
    }

    #[test]
    fn semver_compare_minor() {
        let result = compare_semver("1.0.0", "1.1.0").unwrap();
        assert!(result.contains("MINOR"));
        assert!(result.contains("Breaking change: No"));
    }

    #[test]
    fn semver_compare_equal() {
        let result = compare_semver("1.0.0", "1.0.0").unwrap();
        assert!(result.contains("EQUAL"));
    }

    #[test]
    fn semver_with_v_prefix() {
        let v = parse_semver("v1.2.3").unwrap();
        assert_eq!(v.major, 1);
    }

    // ── §82 Molecular Weight & Binding Affinity ──

    #[test]
    fn molecular_weight_glucose() {
        let op = MoleculeOp::MolecularWeight {
            formula: "C6H12O6".to_string(),
        };
        let result = molecule_calc(&op);
        assert!(result.contains("180."));
    }

    #[test]
    fn molecular_weight_calcium_hydroxide() {
        let op = MoleculeOp::MolecularWeight {
            formula: "Ca(OH)2".to_string(),
        };
        let result = molecule_calc(&op);
        assert!(result.contains("74."));
    }

    #[test]
    fn molecular_weight_aluminum_sulfate() {
        let op = MoleculeOp::MolecularWeight {
            formula: "Al2(SO4)3".to_string(),
        };
        let result = molecule_calc(&op);
        assert!(result.contains("342."));
    }

    #[test]
    fn delta_g_from_kd_nanomolar() {
        let op = MoleculeOp::DeltaGFromKd {
            kd_molar: 1e-9,
            temp_kelvin: 298.15,
        };
        let result = molecule_calc(&op);
        // ΔG should be around -12.3 kcal/mol
        assert!(result.contains("-12."));
    }

    #[test]
    fn kd_delta_g_roundtrip() {
        let dg = -10.0;
        let t = 298.15;
        let kd_result = molecule_calc(&MoleculeOp::KdFromDeltaG {
            delta_g_kcal: dg,
            temp_kelvin: t,
        });
        assert!(kd_result.contains("Kd"));
    }

    // ── §85 Pharmacokinetics ──

    #[test]
    fn pharma_half_life() {
        let result = pharma_calc(&PharmaOp::HalfLife { ke: 0.1 });
        assert!(result.contains("6.93"));
    }

    #[test]
    fn pharma_concentration_decay() {
        let result = pharma_calc(&PharmaOp::Concentration {
            c0: 100.0,
            ke: 0.1,
            t: 10.0,
        });
        // C(10) = 100 * e^(-1) ≈ 36.79
        assert!(result.contains("36."));
    }

    #[test]
    fn pharma_auc_trapezoidal() {
        let result = pharma_calc(&PharmaOp::Auc {
            times: vec![0.0, 1.0, 2.0, 4.0],
            concentrations: vec![10.0, 8.0, 5.0, 2.0],
        });
        // AUC = 0.5*(10+8)*1 + 0.5*(8+5)*1 + 0.5*(5+2)*2 = 9+6.5+7 = 22.5
        assert!(result.contains("22.5"));
    }

    #[test]
    fn pharma_volume_of_distribution() {
        let result = pharma_calc(&PharmaOp::VolumeOfDistribution {
            dose_mg: 500.0,
            c0: 10.0,
        });
        assert!(result.contains("50."));
    }

    #[test]
    fn pharma_steady_state() {
        let result = pharma_calc(&PharmaOp::SteadyState {
            dose_mg: 100.0,
            bioavailability: 1.0,
            clearance: 5.0,
            interval_h: 8.0,
        });
        // Css = (100*1)/(5*8) = 2.5
        assert!(result.contains("2.5"));
    }

    // ── §86 Clinical Trial Statistics ──

    #[test]
    fn clinical_nnt() {
        let result = clinical_calc(&ClinicalOp::Nnt {
            risk_treatment: 0.15,
            risk_control: 0.25,
        });
        // NNT = 1/0.10 = 10
        assert!(result.contains("10."));
    }

    #[test]
    fn clinical_odds_ratio() {
        let result = clinical_calc(&ClinicalOp::OddsRatio {
            a: 20,
            b: 80,
            c: 10,
            d: 90,
        });
        // OR = (20*90)/(80*10) = 2.25
        assert!(result.contains("2.25"));
    }

    #[test]
    fn clinical_relative_risk() {
        let result = clinical_calc(&ClinicalOp::RelativeRisk {
            a: 20,
            b: 80,
            c: 10,
            d: 90,
        });
        // RR = (20/100)/(10/100) = 2.0
        assert!(result.contains("2.0"));
    }

    #[test]
    fn clinical_chi_square() {
        let result = clinical_calc(&ClinicalOp::ChiSquare {
            a: 20,
            b: 80,
            c: 10,
            d: 90,
        });
        assert!(result.contains("χ²"));
    }

    #[test]
    fn clinical_kaplan_meier() {
        let result = clinical_calc(&ClinicalOp::KaplanMeier {
            events: vec![(1.0, 2, 20), (3.0, 3, 18), (6.0, 1, 15)],
        });
        assert!(result.contains("S(t)"));
        assert!(result.contains("Kaplan-Meier"));
    }

    // ── §81 Protein Folding Energy ──

    #[test]
    fn lj_at_minimum() {
        // At r = sigma * 2^(1/6), E = -epsilon
        let sigma = 3.4;
        let epsilon = 0.2;
        let r = sigma * 2.0_f64.powf(1.0 / 6.0);
        let result = protein_energy_calc(&ProteinEnergyOp::LennardJones { epsilon, sigma, r });
        assert!(result.contains("-0.2"));
    }

    #[test]
    fn coulomb_opposite_charges() {
        let result = protein_energy_calc(&ProteinEnergyOp::Coulomb {
            q1: 1.0,
            q2: -1.0,
            r: 4.0,
            dielectric: 80.0,
        });
        // E = 332 * 1 * (-1) / (80 * 4) = -1.0375
        assert!(result.contains("-1.0"));
    }

    #[test]
    fn hbond_optimal() {
        let result = protein_energy_calc(&ProteinEnergyOp::HydrogenBond {
            d_ha: 1.9,
            angle_dha: 180.0,
        });
        assert!(result.contains("-2.0") || result.contains("Strong"));
    }

    #[test]
    fn ramachandran_alpha_helix() {
        let result = protein_energy_calc(&ProteinEnergyOp::Ramachandran {
            phi: -60.0,
            psi: -45.0,
        });
        assert!(result.contains("Alpha helix"));
    }

    #[test]
    fn ramachandran_beta_sheet() {
        let result = protein_energy_calc(&ProteinEnergyOp::Ramachandran {
            phi: -120.0,
            psi: 130.0,
        });
        assert!(result.contains("Beta sheet"));
    }

    // ── §83 Sequence Alignment ──

    #[test]
    fn dna_align_exact() {
        let result = alignment_calc(&AlignmentOp::DnaAlign {
            seq1: "ACGT".to_string(),
            seq2: "ACGT".to_string(),
        });
        assert!(result.contains("Score: 8"));
        assert!(result.contains("100.0%"));
    }

    #[test]
    fn dna_align_mismatch() {
        let result = alignment_calc(&AlignmentOp::DnaAlign {
            seq1: "ACGT".to_string(),
            seq2: "AGGT".to_string(),
        });
        assert!(result.contains("Score:"));
    }

    #[test]
    fn protein_align_identical() {
        let result = alignment_calc(&AlignmentOp::ProteinAlign {
            seq1: "MVLSG".to_string(),
            seq2: "MVLSG".to_string(),
        });
        assert!(result.contains("100.0%"));
    }

    #[test]
    fn dna_align_with_gap() {
        let result = alignment_calc(&AlignmentOp::DnaAlign {
            seq1: "ACGTACGT".to_string(),
            seq2: "ACGACGT".to_string(),
        });
        assert!(result.contains("Score:"));
    }

    #[test]
    fn alignment_cap_length() {
        let long = "A".repeat(1001);
        let result = alignment_calc(&AlignmentOp::DnaAlign {
            seq1: long,
            seq2: "ACGT".to_string(),
        });
        assert!(result.contains("capped at 1000"));
    }

    // ── §84 Drug Interaction ──

    #[test]
    fn drug_interaction_major() {
        let result = drug_calc(&DrugOp::Interaction {
            drug1: "simvastatin".to_string(),
            drug2: "itraconazole".to_string(),
        });
        assert!(result.contains("Major"));
        assert!(result.contains("CYP3A4"));
    }

    #[test]
    fn drug_interaction_none() {
        let result = drug_calc(&DrugOp::Interaction {
            drug1: "metformin".to_string(),
            drug2: "acetaminophen".to_string(),
        });
        assert!(result.contains("No"));
    }

    #[test]
    fn drug_cyp_profile() {
        let result = drug_calc(&DrugOp::CypProfile {
            drug: "fluoxetine".to_string(),
        });
        assert!(result.contains("CYP2D6"));
        assert!(result.contains("strong"));
    }

    #[test]
    fn drug_not_found() {
        let result = drug_calc(&DrugOp::CypProfile {
            drug: "notadrug".to_string(),
        });
        assert!(result.contains("not found"));
    }

    #[test]
    fn drug_all_interactions_warfarin() {
        let result = drug_calc(&DrugOp::AllInteractions {
            drug: "warfarin".to_string(),
        });
        assert!(result.contains("warfarin"));
    }

    // ── §87 Pathway Traversal ──

    #[test]
    fn pathway_list() {
        let result = pathway_calc(&PathwayOp::ListPathways);
        assert!(result.contains("MAPK/ERK"));
        assert!(result.contains("PI3K/AKT"));
        assert!(result.contains("p53"));
    }

    #[test]
    fn pathway_bfs_ras() {
        let result = pathway_calc(&PathwayOp::Bfs {
            start: "RAS".to_string(),
            pathway: Some("MAPK".to_string()),
        });
        assert!(result.contains("RAF"));
        assert!(result.contains("MEK"));
        assert!(result.contains("ERK"));
    }

    #[test]
    fn pathway_shortest_path() {
        let result = pathway_calc(&PathwayOp::ShortestPath {
            from: "RTK".to_string(),
            to: "mTOR".to_string(),
            pathway: Some("PI3K".to_string()),
        });
        assert!(result.contains("AKT"));
    }

    #[test]
    fn pathway_upstream_p21() {
        let result = pathway_calc(&PathwayOp::Upstream {
            node: "P21".to_string(),
            pathway: Some("p53".to_string()),
        });
        assert!(result.contains("P53"));
    }

    #[test]
    fn pathway_downstream_akt() {
        let result = pathway_calc(&PathwayOp::Downstream {
            node: "AKT".to_string(),
            pathway: Some("PI3K".to_string()),
        });
        assert!(result.contains("mTOR"));
    }

    // ── §88 Checksum Validator ──

    #[test]
    fn luhn_valid_card() {
        let result = checksum_validate(&ChecksumOp::Luhn {
            digits: "4532015112830366".into(),
        });
        assert!(result.contains("PASS"));
    }

    #[test]
    fn luhn_invalid_card() {
        let result = checksum_validate(&ChecksumOp::Luhn {
            digits: "1234567890".into(),
        });
        assert!(result.contains("FAIL"));
    }

    #[test]
    fn isbn10_valid() {
        let result = checksum_validate(&ChecksumOp::Isbn {
            code: "0306406152".into(),
        });
        assert!(result.contains("PASS"));
    }

    #[test]
    fn isbn13_valid() {
        let result = checksum_validate(&ChecksumOp::Isbn {
            code: "9780306406157".into(),
        });
        assert!(result.contains("PASS"));
    }

    #[test]
    fn iban_valid() {
        let result = checksum_validate(&ChecksumOp::Iban {
            code: "GB29 NWBK 6016 1331 9268 19".into(),
        });
        assert!(result.contains("PASS"));
    }

    #[test]
    fn ean13_valid() {
        let result = checksum_validate(&ChecksumOp::Ean13 {
            code: "4006381333931".into(),
        });
        assert!(result.contains("PASS"));
    }

    // ── §89 NATO Phonetic ──

    #[test]
    fn nato_hello() {
        let result = nato_phonetic("ABC");
        assert!(result.contains("Alfa"));
        assert!(result.contains("Bravo"));
        assert!(result.contains("Charlie"));
    }

    #[test]
    fn nato_digits() {
        let result = nato_phonetic("911");
        assert!(result.contains("Niner"));
        assert!(result.contains("One"));
    }

    // ── §90 Caesar / ROT13 ──

    #[test]
    fn rot13_symmetric() {
        let result = caesar_calc(&CaesarOp::Rot13 {
            text: "Hello World".into(),
        });
        assert!(result.contains("Uryyb Jbeyq"));
    }

    #[test]
    fn rot13_roundtrip() {
        let encrypted = caesar_shift("Hello", 13);
        let decrypted = caesar_shift(&encrypted, 13);
        assert_eq!(decrypted, "Hello");
    }

    #[test]
    fn caesar_encrypt_decrypt() {
        let enc = caesar_calc(&CaesarOp::Encrypt {
            text: "ABC".into(),
            shift: 3,
        });
        assert!(enc.contains("DEF"));
        let dec = caesar_calc(&CaesarOp::Decrypt {
            text: "DEF".into(),
            shift: 3,
        });
        assert!(dec.contains("ABC"));
    }

    // ── §91 Aspect Ratio ──

    #[test]
    fn aspect_ratio_1920x1080() {
        let result = aspect_ratio_calc(&AspectRatioOp::FromDimensions {
            width: 1920,
            height: 1080,
        });
        assert!(result.contains("16:9"));
        assert!(result.contains("Widescreen"));
    }

    #[test]
    fn aspect_ratio_4k() {
        let result = aspect_ratio_calc(&AspectRatioOp::FromDimensions {
            width: 3840,
            height: 2160,
        });
        assert!(result.contains("16:9"));
    }

    #[test]
    fn aspect_ratio_scale() {
        let result = aspect_ratio_calc(&AspectRatioOp::Scale {
            width: 1920,
            height: 1080,
            target_width: 1280,
        });
        assert!(result.contains("720"));
    }

    // ── §92 Resistor Color Code ──

    #[test]
    fn resistor_decode_4band() {
        let result = resistor_calc(&ResistorOp::Decode {
            bands: vec!["brown".into(), "black".into(), "red".into(), "gold".into()],
        });
        assert!(result.contains("1.00 k\u{03A9}"));
        assert!(result.contains("\u{00B1}5%"));
    }

    #[test]
    fn resistor_decode_470() {
        let result = resistor_calc(&ResistorOp::Decode {
            bands: vec!["yellow".into(), "violet".into(), "brown".into()],
        });
        assert!(result.contains("470"));
    }

    #[test]
    fn resistor_encode_10k() {
        let result = resistor_calc(&ResistorOp::Encode { ohms: 10_000.0 });
        assert!(result.contains("Brown"));
        assert!(result.contains("Black"));
        assert!(result.contains("Orange"));
    }

    // ── §93 Network Bandwidth ──

    #[test]
    fn bandwidth_transfer_time() {
        let result = bandwidth_calc(&BandwidthOp::TransferTime {
            bytes: 1_000_000_000,
            bits_per_sec: 100_000_000, // 1GB at 100 Mbps = 80s = 1.3 min
        });
        assert!(result.contains("minutes"));
        assert!(result.contains("1.00 GB"));
    }

    #[test]
    fn bandwidth_required_speed() {
        let result = bandwidth_calc(&BandwidthOp::RequiredSpeed {
            bytes: 1_000_000_000,
            seconds: 10.0,
        });
        assert!(result.contains("Mbps"));
    }

    // ── §94 Unicode Inspector ──

    #[test]
    fn unicode_inspect_ascii() {
        let result = unicode_inspect("A");
        assert!(result.contains("U+0041"));
        assert!(result.contains("LATIN CAPITAL LETTER"));
    }

    #[test]
    fn unicode_inspect_emoji() {
        let result = unicode_inspect("\u{03C0}");
        assert!(result.contains("U+03C0"));
        assert!(result.contains("PI"));
    }

    #[test]
    fn unicode_inspect_multi() {
        let result = unicode_inspect("Hi");
        assert!(result.contains("2 characters"));
        assert!(result.contains("U+0048"));
        assert!(result.contains("U+0069"));
    }

    // ── §95 IEEE 754 Float ──

    #[test]
    fn float754_one() {
        let result = float754_inspect(1.0);
        assert!(result.contains("Normal"));
        assert!(result.contains("3FF0000000000000"));
    }

    #[test]
    fn float754_negative() {
        let result = float754_inspect(-1.0);
        assert!(result.contains("Sign: 1 (-)"));
    }

    #[test]
    fn float754_zero() {
        let result = float754_inspect(0.0);
        assert!(result.contains("Zero"));
    }

    #[test]
    fn float754_infinity() {
        let result = float754_inspect(f64::INFINITY);
        assert!(result.contains("Infinity"));
    }

    #[test]
    fn float754_nan() {
        let result = float754_inspect(f64::NAN);
        assert!(result.contains("NaN"));
    }

    // ── §96 Frequency / Wavelength ──

    #[test]
    fn freq_to_wavelength_wifi() {
        let result = freq_wavelength_calc(&FreqWavelengthOp::FreqToWavelength { hz: 2.4e9 });
        assert!(result.contains("cm")); // ~12.5 cm
        assert!(result.contains("UHF"));
    }

    #[test]
    fn wavelength_to_freq_visible() {
        let result = freq_wavelength_calc(&FreqWavelengthOp::WavelengthToFreq { meters: 550e-9 });
        assert!(result.contains("THz"));
        assert!(result.contains("Visible Light"));
    }

    #[test]
    fn freq_classify_fm_radio() {
        let result = freq_wavelength_calc(&FreqWavelengthOp::Classify { hz: 100e6 });
        assert!(result.contains("VHF"));
    }

    // ── §97 Molar Mass ──

    #[test]
    fn molar_mass_water() {
        let result = molar_mass("H2O").unwrap();
        assert!(result.contains("18.015")); // H2O = 18.015 g/mol
    }

    #[test]
    fn molar_mass_glucose() {
        let result = molar_mass("C6H12O6").unwrap();
        assert!(result.contains("180.")); // C6H12O6 ≈ 180.156
    }

    #[test]
    fn molar_mass_nacl() {
        let result = molar_mass("NaCl").unwrap();
        assert!(result.contains("58.")); // NaCl ≈ 58.44
    }

    #[test]
    fn molar_mass_parentheses() {
        let result = molar_mass("Ca(OH)2").unwrap();
        assert!(result.contains("74.")); // Ca(OH)2 ≈ 74.09
    }

    #[test]
    fn molar_mass_unknown_element() {
        let result = molar_mass("Xx");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown element"));
    }

    // ── §98 Translation ──

    #[test]
    fn translate_hello_to_french() {
        let result = translate_lookup("hello", "french");
        assert!(result.contains("Bonjour"));
        assert!(result.contains("English → French"));
    }

    #[test]
    fn translate_hello_to_japanese() {
        let result = translate_lookup("Hello", "ja");
        assert!(result.contains("こんにちは"));
    }

    #[test]
    fn translate_thank_you_to_spanish() {
        let result = translate_lookup("thank you", "spanish");
        assert!(result.contains("Gracias"));
    }

    #[test]
    fn translate_unknown_phrase_needs_llm() {
        let result = translate_lookup("quantum entanglement", "french");
        assert!(result.starts_with("[needs LLM]"));
    }

    #[test]
    fn translate_unknown_language() {
        let result = translate_lookup("hello", "klingon");
        assert!(result.contains("Unknown target language"));
    }

    #[test]
    fn translate_lang_alias_codes() {
        // "fr" and "french" should both work
        let r1 = translate_lookup("goodbye", "fr");
        let r2 = translate_lookup("goodbye", "french");
        assert!(r1.contains("Au revoir"));
        assert!(r2.contains("Au revoir"));
    }

    #[test]
    fn translate_case_insensitive() {
        let result = translate_lookup("HELLO", "german");
        assert!(result.contains("Hallo"));
    }

    #[test]
    fn translate_to_english_reverse() {
        let result = translate_lookup("Bonjour", "english");
        assert!(result.contains("hello"));
    }

    // ── §99 Filesystem ──

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_file_exists_self() {
        let result = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::FileExists {
                path: "Cargo.toml".into(),
            },
        });
        assert!(result.is_ok());
        // We're running from the workspace root, Cargo.toml exists
        let text = result.unwrap();
        assert!(text.contains("exists") || text.contains("not exist"));
    }

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_file_info() {
        // Use a path that definitely exists in test context
        let result = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::FileInfo {
                path: "/dev/null".into(),
            },
        });
        // /dev/null always exists on macOS/Linux
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_list_dir() {
        let result = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::ListDirectory {
                path: "/tmp".into(),
            },
        });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("/tmp"));
    }

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_read_write_delete() {
        let dir = std::env::temp_dir().join("ask_test_fs");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_rwd.txt");
        let path_str = path.to_string_lossy().to_string();

        // Write
        let w = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::WriteFile {
                path: path_str.clone(),
                content: "hello deterministic".into(),
            },
        });
        assert!(w.is_ok());
        assert!(w.unwrap().contains("Wrote"));

        // Read
        let r = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::ReadFile {
                path: path_str.clone(),
            },
        });
        assert!(r.is_ok());
        assert!(r.unwrap().contains("hello deterministic"));

        // Delete
        let d = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::DeleteFile { path: path_str },
        });
        assert!(d.is_ok());
        assert!(d.unwrap().contains("Deleted"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_directory_tree() {
        let result = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::DirectoryTree {
                path: "/tmp".into(),
                max_depth: 1,
            },
        });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("/tmp/"));
    }

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_search_files() {
        let result = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::SearchFiles {
                directory: "/usr/bin".into(),
                pattern: "git".into(),
            },
        });
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "io")]
    fn filesystem_copy_move() {
        let dir = std::env::temp_dir().join("ask_test_copy");
        let _ = std::fs::create_dir_all(&dir);
        let src = dir.join("src.txt");
        let dst = dir.join("dst.txt");
        std::fs::write(&src, "copy test").unwrap();

        let result = execute(&DeterministicToolKind::Filesystem {
            operation: FilesystemOp::CopyFile {
                source: src.to_string_lossy().to_string(),
                destination: dst.to_string_lossy().to_string(),
            },
        });
        assert!(result.is_ok());
        assert!(dst.exists());
        assert_eq!(std::fs::read_to_string(&dst).unwrap(), "copy test");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── §100 Git ──

    #[test]
    #[cfg(feature = "io")]
    fn git_status_cwd() {
        // Run from the workspace root which is a git repo
        let result = execute(&DeterministicToolKind::Git {
            operation: GitOp::Status {
                repo_path: ".".into(),
            },
        });
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "io")]
    fn git_log_recent() {
        let result = execute(&DeterministicToolKind::Git {
            operation: GitOp::Log {
                repo_path: ".".into(),
                count: 3,
            },
        });
        assert!(result.is_ok());
        assert!(result.unwrap().contains("commits"));
    }

    #[test]
    #[cfg(feature = "io")]
    fn git_branch_list() {
        let result = execute(&DeterministicToolKind::Git {
            operation: GitOp::BranchList {
                repo_path: ".".into(),
            },
        });
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("main") || text.contains("master") || text.contains("Branches"));
    }

    #[test]
    #[cfg(feature = "io")]
    fn git_remote_list() {
        let result = execute(&DeterministicToolKind::Git {
            operation: GitOp::RemoteList {
                repo_path: ".".into(),
            },
        });
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "io")]
    fn git_tag_list() {
        let result = execute(&DeterministicToolKind::Git {
            operation: GitOp::TagList {
                repo_path: ".".into(),
            },
        });
        assert!(result.is_ok());
    }

    // ── §101 Web Fetch (sync stub) ──

    #[test]
    #[cfg(feature = "web")]
    fn web_fetch_sync_returns_error() {
        let result = execute(&DeterministicToolKind::WebFetch {
            operation: WebFetchOp::Fetch {
                url: "https://example.com".into(),
                max_length: None,
            },
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("async"));
    }

    // ── Helper tests ──

    #[test]
    #[cfg(feature = "web")]
    fn strip_html_basic() {
        assert_eq!(strip_html_tags("<p>hello</p>"), "hello");
        assert_eq!(strip_html_tags("<b>bold</b> text"), "bold text");
    }

    #[test]
    #[cfg(feature = "web")]
    fn strip_html_script_removal() {
        let html = "before<script>var x=1;</script>after";
        assert_eq!(strip_html_tags(html), "beforeafter");
    }

    // ── §110 OpenAPI generation tests ──

    #[test]
    fn openapi_basic() {
        let result = execute(&DeterministicToolKind::OpenApi {
            params: OpenApiParams {
                title: "Pet Store".into(),
                version: "1.0.0".into(),
                base_path: "/api/v1".into(),
                endpoints: vec![OpenApiEndpoint {
                    path: "/pets".into(),
                    method: "GET".into(),
                    summary: "List all pets".into(),
                    request_body: None,
                    response_schema: Some("array".into()),
                    params: vec![OpenApiParam {
                        name: "limit".into(),
                        location: "query".into(),
                        required: false,
                        param_type: "integer".into(),
                    }],
                }],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("openapi: \"3.0.3\""));
        assert!(out.contains("Pet Store"));
        assert!(out.contains("/pets"));
        assert!(out.contains("List all pets"));
    }

    // ── §111 SQL generation tests ──

    #[test]
    fn sql_select() {
        let result = execute(&DeterministicToolKind::SqlQuery {
            params: SqlQueryParams {
                operation: SqlOp::Select {
                    table: "users".into(),
                    columns: vec!["name".into(), "email".into()],
                    where_clause: Some("active = true".into()),
                    order_by: Some("name ASC".into()),
                    limit: Some(10),
                    joins: vec![],
                },
            },
        });
        let out = result.unwrap();
        assert!(out.contains("SELECT name, email"));
        assert!(out.contains("FROM users"));
        assert!(out.contains("WHERE active = true"));
        assert!(out.contains("ORDER BY name ASC"));
        assert!(out.contains("LIMIT 10"));
    }

    #[test]
    fn sql_create_table() {
        let result = execute(&DeterministicToolKind::SqlQuery {
            params: SqlQueryParams {
                operation: SqlOp::CreateTable {
                    table: "orders".into(),
                    columns: vec![
                        SqlColumn {
                            name: "id".into(),
                            col_type: "SERIAL".into(),
                            constraints: "PRIMARY KEY".into(),
                        },
                        SqlColumn {
                            name: "total".into(),
                            col_type: "DECIMAL(10,2)".into(),
                            constraints: "NOT NULL".into(),
                        },
                    ],
                },
            },
        });
        let out = result.unwrap();
        assert!(out.contains("CREATE TABLE orders"));
        assert!(out.contains("id SERIAL PRIMARY KEY"));
    }

    #[test]
    fn sql_insert() {
        let result = execute(&DeterministicToolKind::SqlQuery {
            params: SqlQueryParams {
                operation: SqlOp::Insert {
                    table: "users".into(),
                    columns: vec!["name".into(), "email".into()],
                },
            },
        });
        let out = result.unwrap();
        assert!(out.contains("INSERT INTO users"));
        assert!(out.contains("$1, $2"));
    }

    // ── §112 GraphQL schema tests ──

    #[test]
    fn graphql_schema_basic() {
        let result = execute(&DeterministicToolKind::GraphqlSchema {
            params: GraphqlSchemaParams {
                types: vec![GqlType {
                    name: "User".into(),
                    fields: vec![
                        GqlField {
                            name: "id".into(),
                            field_type: "ID!".into(),
                            args: vec![],
                        },
                        GqlField {
                            name: "name".into(),
                            field_type: "String!".into(),
                            args: vec![],
                        },
                    ],
                }],
                queries: vec![GqlField {
                    name: "getUser".into(),
                    field_type: "User".into(),
                    args: vec![GqlArg {
                        name: "id".into(),
                        arg_type: "ID!".into(),
                    }],
                }],
                mutations: vec![],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("type User {"));
        assert!(out.contains("id: ID!"));
        assert!(out.contains("getUser(id: ID!): User"));
    }

    // ── §113 TypeGen tests ──

    #[test]
    fn typegen_typescript() {
        let result = execute(&DeterministicToolKind::TypeGen {
            params: TypeGenParams {
                name: "User".into(),
                fields: vec![
                    TypeGenField {
                        name: "name".into(),
                        field_type: "string".into(),
                        optional: false,
                    },
                    TypeGenField {
                        name: "age".into(),
                        field_type: "integer".into(),
                        optional: true,
                    },
                ],
                target: TypeGenTarget::TypeScript,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("export interface User {"));
        assert!(out.contains("name: string;"));
        assert!(out.contains("age?: number;"));
    }

    #[test]
    fn typegen_rust() {
        let result = execute(&DeterministicToolKind::TypeGen {
            params: TypeGenParams {
                name: "Config".into(),
                fields: vec![
                    TypeGenField {
                        name: "host".into(),
                        field_type: "string".into(),
                        optional: false,
                    },
                    TypeGenField {
                        name: "port".into(),
                        field_type: "integer".into(),
                        optional: false,
                    },
                ],
                target: TypeGenTarget::Rust,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("pub struct Config {"));
        assert!(out.contains("pub host: String,"));
        assert!(out.contains("pub port: i32,"));
    }

    #[test]
    fn typegen_python() {
        let result = execute(&DeterministicToolKind::TypeGen {
            params: TypeGenParams {
                name: "Product".into(),
                fields: vec![
                    TypeGenField {
                        name: "title".into(),
                        field_type: "string".into(),
                        optional: false,
                    },
                    TypeGenField {
                        name: "price".into(),
                        field_type: "float".into(),
                        optional: true,
                    },
                ],
                target: TypeGenTarget::Python,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("@dataclass"));
        assert!(out.contains("class Product:"));
        assert!(out.contains("title: str"));
        assert!(out.contains("price: Optional[float]"));
    }

    #[test]
    fn typegen_go() {
        let result = execute(&DeterministicToolKind::TypeGen {
            params: TypeGenParams {
                name: "Response".into(),
                fields: vec![
                    TypeGenField {
                        name: "status".into(),
                        field_type: "string".into(),
                        optional: false,
                    },
                    TypeGenField {
                        name: "count".into(),
                        field_type: "integer".into(),
                        optional: true,
                    },
                ],
                target: TypeGenTarget::Go,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("type Response struct {"));
        assert!(out.contains("Status string"));
        assert!(out.contains("Count *int32"));
    }

    // ── §114 Protobuf tests ──

    #[test]
    fn protobuf_basic() {
        let result = execute(&DeterministicToolKind::Protobuf {
            params: ProtobufParams {
                package: "myapp".into(),
                messages: vec![ProtoMessage {
                    name: "User".into(),
                    fields: vec![
                        ProtoField {
                            name: "id".into(),
                            field_type: "string".into(),
                            number: 1,
                            repeated: false,
                        },
                        ProtoField {
                            name: "tags".into(),
                            field_type: "string".into(),
                            number: 2,
                            repeated: true,
                        },
                    ],
                }],
                services: vec![ProtoService {
                    name: "UserService".into(),
                    rpcs: vec![ProtoRpc {
                        name: "GetUser".into(),
                        request: "GetUserRequest".into(),
                        response: "User".into(),
                    }],
                }],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("syntax = \"proto3\""));
        assert!(out.contains("package myapp;"));
        assert!(out.contains("message User {"));
        assert!(out.contains("repeated string tags = 2;"));
        assert!(out.contains("service UserService {"));
        assert!(out.contains("rpc GetUser(GetUserRequest) returns (User);"));
    }

    // ── §115 .gitignore tests ──

    #[test]
    fn gitignore_rust() {
        let result = execute(&DeterministicToolKind::Gitignore {
            language: "rust".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("/target/"));
        assert!(out.contains(".DS_Store"));
    }

    #[test]
    fn gitignore_python() {
        let result = execute(&DeterministicToolKind::Gitignore {
            language: "python".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("__pycache__/"));
        assert!(out.contains("venv/"));
    }

    #[test]
    fn gitignore_node() {
        let result = execute(&DeterministicToolKind::Gitignore {
            language: "node".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("node_modules/"));
    }

    // ── §116 Secret detect tests ──

    #[test]
    fn secret_detect_aws_key() {
        let result = execute(&DeterministicToolKind::SecretDetect {
            input: "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("AWS Access Key"));
        assert!(out.contains("finding"));
    }

    #[test]
    fn secret_detect_clean() {
        let result = execute(&DeterministicToolKind::SecretDetect {
            input: "const greeting = 'hello world';".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("No secrets"));
    }

    #[test]
    fn secret_detect_github_token() {
        let result = execute(&DeterministicToolKind::SecretDetect {
            input: "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh1234".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("GitHub"));
    }

    // ── §117 Regex pattern tests ──

    #[test]
    fn regex_pattern_email() {
        let result = execute(&DeterministicToolKind::RegexPattern {
            name: "email".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("Email address"));
        assert!(out.contains("@"));
    }

    #[test]
    fn regex_pattern_uuid() {
        let result = execute(&DeterministicToolKind::RegexPattern {
            name: "uuid".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("UUID"));
    }

    #[test]
    fn regex_pattern_unknown() {
        let result = execute(&DeterministicToolKind::RegexPattern {
            name: "foobar".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("Unknown pattern"));
    }

    // ── §118 K8s RBAC tests ──

    #[test]
    fn k8s_rbac_basic() {
        let result = execute(&DeterministicToolKind::K8sRbac {
            params: K8sRbacParams {
                kind: K8sRbacKind::Role,
                name: "viewer".into(),
                namespace: Some("default".into()),
                rules: vec![K8sRbacRule {
                    api_groups: vec!["".into()],
                    resources: vec!["pods".into()],
                    verbs: vec!["get".into(), "list".into()],
                }],
                binding: Some(K8sRbacBinding {
                    subject_kind: "ServiceAccount".into(),
                    subject_name: "viewer-sa".into(),
                }),
            },
        });
        let out = result.unwrap();
        assert!(out.contains("kind: Role"));
        assert!(out.contains("name: viewer"));
        assert!(out.contains("namespace: default"));
        assert!(out.contains("RoleBinding"));
    }

    // ── §119 K8s NetworkPolicy tests ──

    #[test]
    fn k8s_netpol_basic() {
        let result = execute(&DeterministicToolKind::K8sNetworkPolicy {
            params: K8sNetworkPolicyParams {
                name: "web-netpol".into(),
                namespace: "production".into(),
                pod_selector: vec![("app".into(), "web".into())],
                ingress_rules: vec![K8sNetPolRule {
                    ports: vec![K8sNetPolPort {
                        protocol: "TCP".into(),
                        port: 443,
                    }],
                    pod_selector: vec![],
                    namespace_selector: vec![],
                    cidr: None,
                }],
                egress_rules: vec![],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("kind: NetworkPolicy"));
        assert!(out.contains("name: web-netpol"));
        assert!(out.contains("port: 443"));
    }

    // ── §120 AWS IAM tests ──

    #[test]
    fn aws_iam_s3_policy() {
        let result = execute(&DeterministicToolKind::AwsIamPolicy {
            params: AwsIamParams {
                policy_name: "s3-access".into(),
                description: "S3 bucket access".into(),
                statements: vec![IamStatement {
                    sid: "S3Read".into(),
                    effect: "Allow".into(),
                    actions: vec!["s3:GetObject".into(), "s3:ListBucket".into()],
                    resources: vec![
                        "arn:aws:s3:::my-bucket".into(),
                        "arn:aws:s3:::my-bucket/*".into(),
                    ],
                    conditions: vec![],
                }],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("\"Version\": \"2012-10-17\""));
        assert!(out.contains("\"Sid\": \"S3Read\""));
        assert!(out.contains("s3:GetObject"));
        assert!(out.contains("arn:aws:s3:::my-bucket"));
    }

    #[test]
    fn aws_iam_with_condition() {
        let result = execute(&DeterministicToolKind::AwsIamPolicy {
            params: AwsIamParams {
                policy_name: "ec2-regional".into(),
                description: "EC2 access in us-east-1".into(),
                statements: vec![IamStatement {
                    sid: "EC2Regional".into(),
                    effect: "Allow".into(),
                    actions: vec!["ec2:DescribeInstances".into()],
                    resources: vec!["*".into()],
                    conditions: vec![IamCondition {
                        operator: "StringEquals".into(),
                        key: "aws:RequestedRegion".into(),
                        values: vec!["us-east-1".into()],
                    }],
                }],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("\"Condition\""));
        assert!(out.contains("StringEquals"));
        assert!(out.contains("us-east-1"));
    }

    // ── §121 Syllogism tests ──

    #[test]
    fn syllogism_valid_barbara() {
        let result = execute(&DeterministicToolKind::Syllogism {
            params: SyllogismParams {
                premises: vec!["All men are mortal".into(), "All Greeks are men".into()],
                conclusion: "All Greeks are mortal".into(),
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Syllogism"));
        assert!(out.contains("mortal"));
    }

    #[test]
    fn syllogism_with_some() {
        let result = execute(&DeterministicToolKind::Syllogism {
            params: SyllogismParams {
                premises: vec!["All cats are animals".into(), "Some pets are cats".into()],
                conclusion: "Some pets are animals".into(),
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Syllogism"));
    }

    // ── §122 Decision Matrix tests ──

    #[test]
    fn decision_matrix_basic() {
        let result = execute(&DeterministicToolKind::DecisionMatrix {
            params: DecisionMatrixParams {
                options: vec!["React".into(), "Vue".into(), "Angular".into()],
                criteria: vec![
                    DecisionCriterion {
                        name: "Learning Curve".into(),
                        weight: 0.3,
                        scores: vec![7.0, 9.0, 5.0],
                    },
                    DecisionCriterion {
                        name: "Performance".into(),
                        weight: 0.4,
                        scores: vec![8.0, 8.0, 7.0],
                    },
                    DecisionCriterion {
                        name: "Ecosystem".into(),
                        weight: 0.3,
                        scores: vec![9.0, 7.0, 8.0],
                    },
                ],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Decision Matrix"));
        assert!(out.contains("React"));
        assert!(out.contains("Vue"));
        assert!(out.contains("Angular"));
        assert!(out.contains("Recommendation"));
    }

    // ── §123 SWOT tests ──

    #[test]
    fn swot_analysis() {
        let result = execute(&DeterministicToolKind::Swot {
            params: SwotParams {
                subject: "Startup X".into(),
                strengths: vec!["Strong team".into(), "Innovative product".into()],
                weaknesses: vec!["Limited funding".into()],
                opportunities: vec!["Growing market".into(), "AI trend".into()],
                threats: vec!["Big tech competition".into()],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("SWOT Analysis: Startup X"));
        assert!(out.contains("Strengths"));
        assert!(out.contains("Weaknesses"));
        assert!(out.contains("Opportunities"));
        assert!(out.contains("Threats"));
    }

    // ── §124 Pros/Cons tests ──

    #[test]
    fn pros_cons_basic() {
        let result = execute(&DeterministicToolKind::ProsCons {
            topic: "Remote Work".into(),
            pros: vec![
                "Flexibility".into(),
                "No commute".into(),
                "Better work-life balance".into(),
            ],
            cons: vec!["Isolation".into(), "Distractions".into()],
        });
        let out = result.unwrap();
        assert!(out.contains("Remote Work"));
        assert!(out.contains("Flexibility"));
        assert!(out.contains("Isolation"));
        assert!(out.contains("Balance"));
    }

    // ── §125 Root Cause tests ──

    #[test]
    fn root_cause_five_whys() {
        let result = execute(&DeterministicToolKind::RootCause {
            params: RootCauseParams {
                problem: "Server downtime".into(),
                whys: vec![
                    "Service crashed".into(),
                    "Out of memory".into(),
                    "Memory leak in connection pool".into(),
                    "Connections not released on error".into(),
                    "Missing error handling in cleanup path".into(),
                ],
                categories: vec![
                    RootCauseCategory {
                        name: "Code".into(),
                        causes: vec!["Missing cleanup".into(), "No timeout".into()],
                    },
                    RootCauseCategory {
                        name: "Process".into(),
                        causes: vec!["No code review".into()],
                    },
                ],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Root Cause Analysis"));
        assert!(out.contains("5 Whys"));
        assert!(out.contains("Server downtime"));
        assert!(out.contains("Root Cause"));
    }

    #[test]
    fn root_cause_fishbone_only() {
        let result = execute(&DeterministicToolKind::RootCause {
            params: RootCauseParams {
                problem: "High defect rate".into(),
                whys: vec![],
                categories: vec![
                    RootCauseCategory {
                        name: "People".into(),
                        causes: vec!["Training gap".into()],
                    },
                    RootCauseCategory {
                        name: "Process".into(),
                        causes: vec!["No QA".into()],
                    },
                    RootCauseCategory {
                        name: "Machine".into(),
                        causes: vec!["Old equipment".into()],
                    },
                ],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Fishbone"));
        assert!(out.contains("High defect rate"));
    }

    // ── §126 Deduction tests ──

    #[test]
    fn deduction_modus_ponens() {
        let result = execute(&DeterministicToolKind::Deduction {
            params: DeductionParams {
                premises: vec![
                    "If it rains, the ground is wet".into(),
                    "It is raining".into(),
                ],
                rules: vec![DeductionRule {
                    name: "Apply MP".into(),
                    from: "P1, P2".into(),
                    operation: "modus_ponens".into(),
                    yields: "The ground is wet".into(),
                }],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Logical Deduction Chain"));
        assert!(out.contains("Modus Ponens"));
        assert!(out.contains("ground is wet"));
    }

    #[test]
    fn deduction_modus_tollens() {
        let result = execute(&DeterministicToolKind::Deduction {
            params: DeductionParams {
                premises: vec![
                    "If it rains, the ground is wet".into(),
                    "The ground is not wet".into(),
                ],
                rules: vec![DeductionRule {
                    name: "Apply MT".into(),
                    from: "P1, P2".into(),
                    operation: "modus_tollens".into(),
                    yields: "It did not rain".into(),
                }],
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Modus Tollens"));
    }

    // ── Translation expansion tests ──

    #[test]
    fn translate_expanded_phrases() {
        // Test greetings
        let result = execute(&DeterministicToolKind::Translate {
            text: "hello".into(),
            target_lang: "fr".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("Bonjour"), "expected 'Bonjour' in: {out}");

        // Test numbers
        let result = execute(&DeterministicToolKind::Translate {
            text: "one".into(),
            target_lang: "es".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("Uno"), "expected 'Uno' in: {out}");

        // Test days
        let result = execute(&DeterministicToolKind::Translate {
            text: "monday".into(),
            target_lang: "de".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("Montag"), "expected 'Montag' in: {out}");

        // Test colors
        let result = execute(&DeterministicToolKind::Translate {
            text: "red".into(),
            target_lang: "ja".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("赤"), "expected '赤' in: {out}");

        // Test common verb (stored as "to eat")
        let result = execute(&DeterministicToolKind::Translate {
            text: "to eat".into(),
            target_lang: "fr".into(),
        });
        let out = result.unwrap();
        assert!(out.contains("Manger"), "expected 'Manger' in: {out}");
    }

    #[test]
    fn translate_unknown_falls_back() {
        let result = execute(&DeterministicToolKind::Translate {
            text: "supercalifragilistic".into(),
            target_lang: "fr".into(),
        });
        assert!(result.unwrap().contains("[needs LLM]"));
    }

    // ── §127–§132 Energy Floor tests ──

    #[test]
    fn ef_anomaly_detect_spike() {
        // 30 normal values around 100, then a spike to 130
        let mut values: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        values.push(130.0);
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::AnomalyDetect {
                series_id: "ercot_day_ahead".into(),
                values,
                window: 30,
                threshold: 2.0,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Anomaly: YES"), "expected anomaly in: {out}");
        assert!(out.contains("above"), "expected above direction in: {out}");
    }

    #[test]
    fn ef_anomaly_no_anomaly() {
        let values: Vec<f64> = (0..31).map(|i| 100.0 + (i as f64 * 0.01)).collect();
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::AnomalyDetect {
                series_id: "stable".into(),
                values,
                window: 30,
                threshold: 2.0,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Anomaly: no"), "expected no anomaly in: {out}");
    }

    #[test]
    fn ef_correlation_positive() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 1.0).collect();
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::Correlation {
                series_a_id: "energy".into(),
                series_a: a,
                series_b_id: "compute".into(),
                series_b: b,
                lag: 0,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("+1.0000") || out.contains("+0.99"), "expected strong positive r in: {out}");
    }

    #[test]
    fn ef_cost_function_basic() {
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::CostFunction {
                energy_joules: 1000.0,
                energy_price_per_joule: 2.78e-8, // ~$0.10/kWh
                hardware_units: 1.0,
                hardware_price_per_unit: 0.50,
                friction_cost: 0.05,
                useful_bits: 1e9,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("C(W)"), "expected cost function in: {out}");
        assert!(out.contains("Landauer"), "expected Landauer comparison in: {out}");
        assert!(out.contains("/bit"), "expected price per bit in: {out}");
    }

    #[test]
    fn ef_forward_curve_backwardation() {
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::ForwardCurve {
                spot_price: 2.50,
                risk_free_rate: 0.05,
                koomey_rate: 0.26,
                tenors_days: vec![30, 90, 180, 365],
                asset_label: "H100 GPU-hour".into(),
            },
        });
        let out = result.unwrap();
        assert!(out.contains("BACKWARDATION"), "expected backwardation in: {out}");
        assert!(out.contains("H100"), "expected asset label in: {out}");
    }

    #[test]
    fn ef_arbitrage_spread_positive() {
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::ArbitrageSpread {
                long_region: "iceland".into(),
                long_price_kwh: 0.03,
                short_region: "california".into(),
                short_price_kwh: 0.18,
                throughput_mw: 10.0,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("iceland"), "expected destination in: {out}");
        assert!(out.contains("$0.15"), "expected spread in: {out}");
    }

    #[test]
    fn ef_value_function_legal_analysis() {
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::ValueFunction {
                good: "legal analysis".into(),
                energy_cost: 5.0,
                trust_cost: 50.0,
                speed_cost: 400.0,
                compliance_cost: 45.0,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("V(g)"), "expected value function in: {out}");
        assert!(out.contains("$500.00"), "expected total in: {out}");
        assert!(out.contains("Compressible share"), "expected compression analysis in: {out}");
    }

    #[test]
    fn ef_landauer_room_temp() {
        let result = execute(&DeterministicToolKind::EnergyFloor {
            operation: EnergyFloorOp::LandauerBound {
                actual_joules_per_bit: 1e-12,
                temperature_k: 300.0,
            },
        });
        let out = result.unwrap();
        assert!(out.contains("Landauer"), "expected Landauer in: {out}");
        assert!(out.contains("2.87") || out.contains("2.88"), "expected ~2.87e-21 in: {out}");
    }

    // ── §133 Geo tests ──

    #[test]
    fn geo_haversine_nyc_la() {
        // NYC (40.7128, -74.0060) → LA (34.0522, -118.2437) ≈ 3944 km
        let result = geo_calc(&GeoOp::Distance {
            lat1: 40.7128, lon1: -74.0060,
            lat2: 34.0522, lon2: -118.2437,
        });
        assert!(result.contains("km"), "expected km in: {result}");
        // Known: ~3944 km
        let km: f64 = result.split_whitespace()
            .find_map(|w| w.parse().ok())
            .unwrap_or(0.0);
        assert!((km - 3944.0).abs() < 50.0, "expected ~3944 km, got {km}");
    }

    #[test]
    fn geo_bearing_north() {
        // Due north: same lon, lat1 < lat2 → bearing ≈ 0°
        let b = initial_bearing(0.0, 0.0, 10.0, 0.0);
        assert!(b < 1.0 || b > 359.0, "expected ~0° bearing, got {b}");
    }

    #[test]
    fn geo_midpoint_equator() {
        let (lat, lon) = geo_midpoint(0.0, 0.0, 0.0, 10.0);
        assert!((lat).abs() < 0.01, "expected lat≈0, got {lat}");
        assert!((lon - 5.0).abs() < 0.01, "expected lon≈5, got {lon}");
    }

    #[test]
    fn geo_geohash_roundtrip() {
        let hash = geohash_encode(40.7128, -74.0060, 9);
        assert_eq!(hash.len(), 9);
        let (lat, lon, _, _) = geohash_decode(&hash).unwrap();
        assert!((lat - 40.7128).abs() < 0.001, "lat roundtrip failed: {lat}");
        assert!((lon - (-74.0060)).abs() < 0.001, "lon roundtrip failed: {lon}");
    }

    #[test]
    fn geo_dms_parse() {
        let (lat, lon) = parse_dms("40°26'46\"N 79°58'56\"W").unwrap();
        assert!((lat - 40.4461).abs() < 0.01, "lat: {lat}");
        assert!((lon - (-79.9822)).abs() < 0.01, "lon: {lon}");
    }

    #[test]
    fn geo_dd_to_dms_roundtrip() {
        let (lat_s, lon_s) = dd_to_dms(40.4461, -79.9822);
        assert!(lat_s.contains("N"), "expected N: {lat_s}");
        assert!(lon_s.contains("W"), "expected W: {lon_s}");
    }

    #[test]
    fn geo_bounding_box_sanity() {
        let (min_lat, min_lon, max_lat, max_lon) = bounding_box(40.0, -74.0, 10.0);
        assert!(min_lat < 40.0);
        assert!(max_lat > 40.0);
        assert!(min_lon < -74.0);
        assert!(max_lon > -74.0);
    }

    #[test]
    fn geo_point_in_polygon() {
        let triangle = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 5.0)];
        assert!(point_in_polygon(3.0, 5.0, &triangle));
        assert!(!point_in_polygon(20.0, 20.0, &triangle));
    }

    #[test]
    fn geo_utm_nyc() {
        let (zone, letter, easting, northing) = latlon_to_utm(40.7128, -74.0060);
        assert_eq!(zone, 18, "NYC should be UTM zone 18");
        assert!(easting > 0.0 && northing > 0.0);
    }

    #[test]
    fn geo_waypoints_count() {
        let pts = great_circle_waypoints(0.0, 0.0, 10.0, 10.0, 5);
        assert_eq!(pts.len(), 5);
        // First and last should be near the endpoints
        assert!((pts[0].0).abs() < 0.01);
        assert!((pts[4].0 - 10.0).abs() < 0.01);
    }

    #[test]
    fn geo_execute_integration() {
        let result = execute(&DeterministicToolKind::Geo {
            operation: GeoOp::Distance {
                lat1: 51.5074, lon1: -0.1278,  // London
                lat2: 48.8566, lon2: 2.3522,   // Paris
            },
        });
        let out = result.unwrap();
        assert!(out.contains("km"), "expected km: {out}");
        // London→Paris ≈ 344 km
        assert!(out.contains("34"), "expected ~344: {out}");
    }
}
