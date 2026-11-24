//! Team-only Volt communication protocol, matching Python's
//! `volt_teamplay_v1` JSON contract.
use crate::state::TeamplayMessage;

const PROTOCOL: &str = "volt_teamplay_v1";

fn field_start<'a>(source: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\"", key);
    let start = source.find(&needle)? + needle.len();
    let rest = source[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    Some(rest)
}

fn string_field(source: &str, key: &str) -> Option<String> {
    let rest = field_start(source, key)?;
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

fn number_field(source: &str, key: &str) -> Option<f32> {
    let rest = field_start(source, key)?;
    let end = rest.find(|c: char| !(c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))).unwrap_or(rest.len());
    let value = rest[..end].parse::<f32>().ok()?;
    value.is_finite().then_some(value)
}

fn optional_number_field(source: &str, key: &str) -> Option<Option<f32>> {
    let Some(rest) = field_start(source, key) else { return Some(None); };
    if rest.starts_with("null") { Some(None) }
    else { number_field(source, key).map(Some) }
}

fn bool_field(source: &str, key: &str) -> Option<bool> {
    let rest = field_start(source, key)?;
    if rest.starts_with("true") { Some(true) }
    else if rest.starts_with("false") { Some(false) }
    else { None }
}

fn allowed_status(status: &str) -> bool {
    matches!(
        status,
        "SHOT_FOUND" | "ANY_SHOT_FOUND" | "WALL_PASS" | "SHOT_ABANDONED"
            | "LOW_BOOST" | "NO_SHOT" | "DEFEND" | "SHADOW"
    )
}

/// Decode and validate a teammate message. Invalid, enemy, self-originated,
/// wrong-protocol, and malformed messages are ignored.
pub fn decode(
    sender: usize,
    sender_team: u32,
    our_team: u32,
    our_index: usize,
    content: &[u8],
    received_at: f32,
) -> Option<(usize, TeamplayMessage)> {
    if sender_team != our_team || sender == our_index { return None; }
    let text = std::str::from_utf8(content).ok()?;
    if string_field(text, "protocol")?.as_str() != PROTOCOL { return None; }
    let status = string_field(text, "status")?;
    if !allowed_status(&status) { return None; }
    let score = number_field(text, "score")?;
    let contact_time = number_field(text, "contact_time")?;
    if !(0.0..=100.0).contains(&score) || !(0.0..=10.0).contains(&contact_time) {
        return None;
    }
    let cover_released = bool_field(text, "cover_released")?;
    let cover_owner = optional_number_field(text, "cover_owner")?
        .map(|value| value.max(0.0) as usize);
    let shot_type = string_field(text, "shot_type").unwrap_or_else(|| status.clone());
    Some((sender, TeamplayMessage {
        status,
        shot_type,
        score,
        contact_time,
        received_at,
        cover_owner,
        cover_released,
        receive_x: optional_number_field(text, "receive_x")?,
        receive_y: optional_number_field(text, "receive_y")?,
        receive_dir_x: optional_number_field(text, "receive_dir_x")?,
        receive_dir_y: optional_number_field(text, "receive_dir_y")?,
        contact_x: optional_number_field(text, "contact_x")?,
        contact_y: optional_number_field(text, "contact_y")?,
    }))
}

/// Encode the exact JSON fields consumed by Python Volt. The RLBot envelope
/// supplies the sender index/team, so they are not duplicated in the payload.
pub fn encode(_index: usize, _team: u32, message: &TeamplayMessage) -> Option<Vec<u8>> {
    fn opt(value: Option<f32>) -> String {
        value.filter(|v| v.is_finite())
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "null".to_owned())
    }
    let status = if allowed_status(&message.status) { &message.status } else { "NO_SHOT" };
    let shot_type = if matches!(status, "SHOT_FOUND" | "ANY_SHOT_FOUND" | "WALL_PASS") {
        match message.shot_type.as_str() {
            "GROUND" | "JUMP" | "DOUBLE_JUMP" | "AERIAL" => message.shot_type.as_str(),
            _ => status,
        }
    } else { "NO_SHOT" };
    let cover_owner = message.cover_owner.map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_owned());
    let content = format!(
        "{{\"protocol\":\"{}\",\"status\":\"{}\",\"shot_type\":\"{}\",\"score\":{:.1},\"contact_time\":{:.2},\"cover_owner\":{},\"cover_released\":{},\"receive_x\":{},\"receive_y\":{},\"receive_dir_x\":{},\"receive_dir_y\":{},\"contact_x\":{},\"contact_y\":{}}}",
        PROTOCOL, status, shot_type, message.score.clamp(0.0, 100.0),
        message.contact_time.clamp(0.0, 10.0), cover_owner,
        message.cover_released, opt(message.receive_x), opt(message.receive_y),
        opt(message.receive_dir_x), opt(message.receive_dir_y),
        opt(message.contact_x), opt(message.contact_y)
    );
    Some(content.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message() -> TeamplayMessage {
        TeamplayMessage {
            status: "SHOT_FOUND".into(), shot_type: "GROUND".into(),
            score: 82.5, contact_time: 1.25, received_at: 3.0,
            cover_owner: Some(2), cover_released: true,
            receive_x: Some(120.0), receive_y: Some(800.0),
            receive_dir_x: Some(0.0), receive_dir_y: Some(1.0),
            contact_x: Some(100.0), contact_y: Some(300.0),
        }
    }

    #[test]
    fn python_protocol_round_trip_preserves_teamplay_fields() {
        let encoded = encode(1, 0, &message()).unwrap();
        let (sender, decoded) = decode(1, 0, 0, 2, &encoded, 4.0).unwrap();
        assert_eq!(sender, 1);
        assert_eq!(decoded.status, "SHOT_FOUND");
        assert_eq!(decoded.shot_type, "GROUND");
        assert_eq!(decoded.cover_owner, Some(2));
        assert_eq!(decoded.receive_x, Some(120.0));
        assert_eq!(decoded.received_at, 4.0);
    }

    #[test]
    fn protocol_accepts_messages_without_optional_receive_plan() {
        let mut value = message();
        value.status = "NO_SHOT".into();
        value.cover_released = false;
        value.receive_x = None;
        value.receive_y = None;
        value.receive_dir_x = None;
        value.receive_dir_y = None;
        value.contact_x = None;
        value.contact_y = None;
        let encoded = encode(1, 0, &value).unwrap();
        assert!(decode(1, 0, 0, 2, &encoded, 4.0).is_some());
    }

    #[test]
    fn protocol_rejects_enemy_invalid_status_and_out_of_range_values() {
        let encoded = encode(1, 0, &message()).unwrap();
        assert!(decode(1, 1, 0, 2, &encoded, 4.0).is_none());
        let invalid = std::str::from_utf8(&encoded).unwrap().replace("SHOT_FOUND", "INVALID");
        assert!(decode(1, 0, 0, 2, invalid.as_bytes(), 4.0).is_none());
        let invalid = std::str::from_utf8(&encoded).unwrap()
            .replace("\"score\":82.5", "\"score\":101");
        assert!(decode(1, 0, 0, 2, invalid.as_bytes(), 4.0).is_none());
    }
}
